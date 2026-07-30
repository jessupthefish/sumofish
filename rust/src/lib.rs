//! SumoFish hot path in Rust.
//!
//! Scope, and the order it lands in:
//!
//! * **Stage A** board, FEN, move generation, make. *Done, three oracles green.*
//! * **Stage B** terminal detection: mate, stalemate, insufficient material,
//!   fifty-move, threefold repetition over an undo stack. *In progress.*
//! * **Stage C** the tree: PUCT selection, virtual loss, backup. Python keeps
//!   only the network, called once per batch of leaves.
//! * **Stage D** tokenization straight into the tensor buffer.
//!
//! B and C land together: repetition needs history, and the history
//! representation IS the tree layout. See `position.rs`.
//!
//! The governing rule for all four stages is that this is a **faithful port**.
//! Each stage must reproduce the Python implementation exactly, verified by an
//! identity test, before any behaviour changes. Known defects in the Python
//! search (leaf duplication, virtual loss corrupting Q rather than only N, no
//! mate distance) are deliberately reproduced here and fixed afterwards as
//! separate, separately-measured changes. Fixing them during the port would
//! destroy the only cheap way to prove the port is correct, which is that the
//! two implementations agree bit for bit.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

pub mod action_space;
pub mod attacks;
pub mod board;
pub mod movegen;
pub mod position;
pub mod softmax;
pub mod tokenizer;
pub mod tree;

use board::Board;

/// A chess position. Mirrors the parts of `chess.Board` the search actually
/// uses; it is not a general-purpose replacement for `python-chess`.
#[pyclass(name = "Board")]
pub struct PyBoard {
    inner: Board,
}

#[pymethods]
impl PyBoard {
    #[new]
    #[pyo3(signature = (fen = None))]
    fn new(fen: Option<&str>) -> PyResult<Self> {
        let inner = match fen {
            None => Board::starting_position(),
            Some(f) => Board::from_fen(f).map_err(PyValueError::new_err)?,
        };
        Ok(PyBoard { inner })
    }

    fn fen(&self) -> String {
        self.inner.to_fen()
    }

    /// Legal moves as UCI strings, in `python-chess` generation order.
    /// The ordering is part of the contract; see `movegen`'s module docs.
    fn legal_moves(&self) -> Vec<String> {
        movegen::generate_legal_moves(&self.inner)
            .iter()
            .map(|m| m.uci())
            .collect()
    }

    /// Apply a UCI move. Raises on an unparseable string; legality is the
    /// caller's business, exactly as with `chess.Board.push`.
    fn push_uci(&mut self, uci: &str) -> PyResult<()> {
        let mv = movegen::Move::from_uci(uci)
            .ok_or_else(|| PyValueError::new_err(format!("bad uci move '{}'", uci)))?;
        self.inner.push(mv);
        Ok(())
    }

    /// The FEN with the ep square echoed as stored, i.e. python-chess's
    /// `fen(en_passant="fen")`.
    fn fen_raw(&self) -> String {
        self.inner.to_fen_raw()
    }

    fn has_legal_en_passant(&self) -> bool {
        self.inner.has_legal_en_passant()
    }

    fn is_check(&self) -> bool {
        match self.inner.king(self.inner.turn) {
            Some(k) => self.inner.is_attacked_by(self.inner.turn.other(), k),
            None => false,
        }
    }

    #[getter]
    fn turn(&self) -> bool {
        // True for White, matching python-chess's chess.WHITE == True.
        self.inner.turn == board::Color::White
    }

    #[getter]
    fn halfmove_clock(&self) -> u32 {
        self.inner.halfmove_clock
    }

    #[getter]
    fn fullmove_number(&self) -> u32 {
        self.inner.fullmove_number
    }

    fn __repr__(&self) -> String {
        format!("Board('{}')", self.inner.to_fen())
    }
}

/// A board plus the history a tree search needs. `push`/`pop`, repetition, and
/// the terminal test. Mirrors what `chessgpu/rules.py` asks of `chess.Board`.
#[pyclass(name = "Position")]
pub struct PyPosition {
    inner: position::Position,
}

#[pymethods]
impl PyPosition {
    #[new]
    #[pyo3(signature = (fen = None))]
    fn new(fen: Option<&str>) -> PyResult<Self> {
        let b = match fen {
            None => Board::starting_position(),
            Some(f) => Board::from_fen(f).map_err(PyValueError::new_err)?,
        };
        Ok(PyPosition { inner: position::Position::new(b) })
    }

    fn fen(&self) -> String {
        self.inner.board.to_fen()
    }

    fn legal_moves(&self) -> Vec<String> {
        movegen::generate_legal_moves(&self.inner.board)
            .iter()
            .map(|m| m.uci())
            .collect()
    }

    fn push_uci(&mut self, uci: &str) -> PyResult<()> {
        let mv = movegen::Move::from_uci(uci)
            .ok_or_else(|| PyValueError::new_err(format!("bad uci move '{}'", uci)))?;
        self.inner.push(mv);
        Ok(())
    }

    fn pop(&mut self) {
        self.inner.pop();
    }

    #[pyo3(signature = (count = 3))]
    fn is_repetition(&self, count: u32) -> bool {
        self.inner.is_repetition(count)
    }

    fn is_insufficient_material(&self) -> bool {
        self.inner.board.is_insufficient_material()
    }

    fn is_check(&self) -> bool {
        self.inner.board.is_check()
    }

    /// Value to the side to move if the game ends here, else None.
    fn terminal_value(&self) -> Option<f64> {
        self.inner.terminal_value()
    }

    /// The 77 tokens for the current position, as bytes.
    fn tokens(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyBytes>> {
        let tokens = self.inner.board.tokenize().map_err(PyValueError::new_err)?;
        Ok(pyo3::types::PyBytes::new(py, &tokens).unbind())
    }

    #[getter]
    fn ply(&self) -> usize {
        self.inner.ply()
    }

    #[getter]
    fn halfmove_clock(&self) -> u32 {
        self.inner.board.halfmove_clock
    }
}

/// Bridges the Rust search to a Python callable supplying the networks.
///
/// The contract is `evaluate(fens, actions) -> (priors, values)`, positional
/// throughout: `actions[i]` is one action index per legal move of `fens[i]` in
/// move order, and `priors[i]` must come back in that same order. This works
/// only because Stage A proved the two implementations generate moves in the
/// same order over 100k positions; a set-equal but differently-ordered generator
/// would silently attach every prior to the wrong move, and every move played
/// would still be legal.
///
/// Passing the action indices is what keeps `python-chess` out of the callback.
/// See `tree::Evaluator` for the measurement that forced it, and for why the
/// softmax has to stay on the numpy side.
struct PyEvaluator<'a, 'py> {
    func: &'a Bound<'py, PyAny>,
}

impl tree::Evaluator for PyEvaluator<'_, '_> {
    fn evaluate(
        &mut self,
        fens: &[String],
        actions: &[Vec<i32>],
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
        let res = self
            .func
            .call1((fens.to_vec(), actions.to_vec()))
            .map_err(|e| format!("evaluate() raised: {}", e))?;
        res.extract::<(Vec<Vec<f64>>, Vec<f64>)>()
            .map_err(|e| format!("evaluate() must return (list[list[float]], list[float]): {}", e))
    }
}

/// MCTS over a value net, with a policy net supplying priors. A faithful port of
/// `chessgpu/mcts.py`; see `tree.rs` for which known defects are reproduced on
/// purpose and why.
#[pyclass(name = "Mcts")]
pub struct PyMcts {
    inner: tree::Mcts,
}

#[pymethods]
impl PyMcts {
    #[new]
    #[pyo3(signature = (c_puct = 2.0, c_puct_base = Some(19652.0), c_puct_init = 1.25, fpu = -0.2, batch = 1, reuse = false, dedup = false, mate_distance = false, vloss_fix = false))]
    #[allow(clippy::too_many_arguments)] // it mirrors mcts.py's constructor
    fn new(
        c_puct: f64,
        c_puct_base: Option<f64>,
        c_puct_init: f64,
        fpu: f64,
        batch: usize,
        reuse: bool,
        dedup: bool,
        mate_distance: bool,
        vloss_fix: bool,
    ) -> Self {
        let mut inner = tree::Mcts::new(c_puct, c_puct_base, c_puct_init, fpu, batch);
        inner.reuse = reuse;
        inner.dedup = dedup;
        inner.mate_distance = mate_distance;
        inner.vloss_fix = vloss_fix;
        PyMcts { inner }
    }

    /// Forget the retained tree. Between games, never within one.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Simulations inherited from the previous move's tree.
    #[getter]
    fn reused(&self) -> i64 {
        self.inner.reused
    }

    /// Search from `position` and return `[(uci, visits), ...]` in child order.
    ///
    /// `position` must carry the game's history: repetition depends on the moves
    /// before the root, so a Position built from a bare FEN will disagree with
    /// Python about threefold draws. It is cloned, so the caller's object is
    /// left untouched.
    ///
    /// `max_seconds`, when given, stops early once elapsed (checked between
    /// batches). `chessgpu/rust_mcts.py::search` always passes this
    /// positionally, even when it is `None` -- this parameter existing at all,
    /// with this name, in this position, is load-bearing for that call site.
    #[pyo3(signature = (position, simulations, evaluate, max_seconds = None))]
    fn search(
        &mut self,
        position: &PyPosition,
        simulations: u32,
        evaluate: &Bound<'_, PyAny>,
        max_seconds: Option<f64>,
    ) -> PyResult<Vec<(String, i64)>> {
        let mut pos = position.inner.clone();
        let mut ev = PyEvaluator { func: evaluate };
        let root = self
            .inner
            .search(&mut pos, simulations, &mut ev, max_seconds)
            .map_err(PyRuntimeError::new_err)?;
        Ok(self.inner.root_visits(root))
    }

    /// The root's proven mate, if the search found one: `(is_win, plies)`.
    fn root_proven(&self) -> Option<(bool, u16)> {
        if self.inner.nodes.is_empty() {
            return None;
        }
        self.inner.root_proven(0)
    }

    /// The move the search would play: most visits, ties to the first child.
    fn best_move(&self) -> Option<String> {
        if self.inner.nodes.is_empty() {
            return None;
        }
        self.inner.best_move(0)
    }

    #[getter]
    fn evaluations(&self) -> u64 {
        self.inner.evaluations
    }

    /// Distinct positions actually sent to the evaluator. With dedup off this
    /// equals `evaluations`; with it on, the gap IS the duplicated work.
    #[getter]
    fn unique_evaluations(&self) -> u64 {
        self.inner.unique_evaluations
    }

    #[getter]
    fn node_count(&self) -> usize {
        self.inner.nodes.len()
    }
}

/// `_softmax_over_legal` in Rust, exposed so the bit-exactness test can compare
/// it against the Python original directly rather than only through a search.
#[pyfunction]
fn softmax_over_legal(row: Vec<f32>, actions: Vec<i32>) -> PyResult<Vec<f64>> {
    if actions.is_empty() {
        return Err(PyValueError::new_err("no legal moves"));
    }
    Ok(softmax::softmax_over_legal(&row, &actions))
}

/// numpy's pairwise summation, exposed for the same reason.
#[pyfunction]
fn pairwise_sum_f32(values: Vec<f32>) -> f32 {
    softmax::pairwise_sum_f32(&values)
}

#[pyfunction]
fn pairwise_sum_f64(values: Vec<f64>) -> f64 {
    softmax::pairwise_sum_f64(&values)
}

/// The 77 tokens for a FEN, identical to `tokenizer.tokenize_board`.
///
/// Returned as `bytes` rather than a list of ints so the caller can go straight
/// to `np.frombuffer(..., dtype=np.uint8)` with no per-element Python objects --
/// which is the entire point of doing this in Rust.
#[pyfunction]
fn tokenize(py: Python<'_>, fen: &str) -> PyResult<Py<pyo3::types::PyBytes>> {
    let b = Board::from_fen(fen).map_err(PyValueError::new_err)?;
    let tokens = b.tokenize().map_err(PyValueError::new_err)?;
    Ok(pyo3::types::PyBytes::new(py, &tokens).unbind())
}

/// The 77 tokens for a whole batch, concatenated: `len(fens) * 77` bytes.
///
/// One boundary crossing and one allocation for the batch, so Python does
/// `np.frombuffer(buf, np.uint8).reshape(-1, 77)` and is done. This is what the
/// search actually wants; `tokenize` above exists for the verification harness.
#[pyfunction]
fn tokenize_batch(py: Python<'_>, fens: Vec<String>) -> PyResult<Py<pyo3::types::PyBytes>> {
    let mut out = Vec::with_capacity(fens.len() * 77);
    for fen in &fens {
        let b = Board::from_fen(fen).map_err(PyValueError::new_err)?;
        out.extend_from_slice(&b.tokenize().map_err(PyValueError::new_err)?);
    }
    Ok(pyo3::types::PyBytes::new(py, &out).unbind())
}

/// Nodes at exactly `depth` plies from `fen`.
#[pyfunction]
fn perft(fen: &str, depth: u32) -> PyResult<u64> {
    let b = Board::from_fen(fen).map_err(PyValueError::new_err)?;
    Ok(movegen::perft(&b, depth))
}

/// Round-trip a FEN through the Rust board. Exposed so the differential test
/// can check parsing and formatting independently of move generation, which is
/// the part that is still a stub.
#[pyfunction]
fn fen_roundtrip(fen: &str) -> PyResult<String> {
    let b = Board::from_fen(fen).map_err(PyValueError::new_err)?;
    Ok(b.to_fen())
}

#[pymodule]
fn sumofish_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBoard>()?;
    m.add_class::<PyPosition>()?;
    m.add_class::<PyMcts>()?;
    m.add_function(wrap_pyfunction!(perft, m)?)?;
    m.add_function(wrap_pyfunction!(fen_roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(softmax_over_legal, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize_batch, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_sum_f32, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_sum_f64, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
