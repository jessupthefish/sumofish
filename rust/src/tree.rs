//! MCTS: the tree, PUCT selection, virtual loss, backup.
//!
//! A **faithful** port of `chessgpu/mcts.py`. Every arithmetic expression is in
//! the same form and the same order as the Python, because the acceptance test
//! is a byte-identical root visit vector rather than "looks about right".
//!
//! # Defects reproduced on purpose
//!
//! Three known problems in the Python search are preserved here, behind flags,
//! each OFF by default so the faithful port stays byte-identical unless asked:
//!
//! 1. **No leaf deduplication.** Two descents that reach the same unexpanded
//!    leaf produce two `pending` entries and two network rows. This is the
//!    mechanism behind the measured 98.6% duplicated work at batch 1024.
//!    Fixed by `dedup` (see `simulate_batch`), off by default.
//! 2. **Virtual loss corrupts Q, not just N.** `backup(path, 1.0)` used to add
//!    straight to `value_sum`, so the exploitation term was distorted, where
//!    Lc0 keeps a separate in-flight counter that only affects the PUCT
//!    denominator. Fixed by `vloss_fix` (see `Node::pending`,
//!    `apply_virtual_loss`, `remove_virtual_loss`, `select_child`), off by
//!    default.
//! 3. **No mate distance.** A mate in 2 and a mate in 14 back up identically.
//!    Fixed by `mate_distance` (see `prove`, `select_child`, `best_move`), off
//!    by default.
//!
//! Turning any of them on here is a genuine behaviour change and will fail the
//! identity test on purpose -- that is what the flag is for. Leaving all three
//! off reproduces the original faithful port exactly. Each is measured on its
//! own, separately, against the pair-match harness. Do not "improve" this file
//! by changing a default.
//!
//! # Sign convention
//!
//! Every value is from the perspective of the side to move AT THAT NODE. A
//! parent scoring a child therefore uses `1 - child.q`, and backup flips at
//! every level. Getting this wrong once made checkmate the worst-scoring move on
//! the board; see the Python file's docstring.
//!
//! # Arena, not pointers
//!
//! Nodes live in a `Vec` and children are indices. A pointer tree needs
//! `Rc<RefCell<..>>` in safe Rust, which costs a refcount and a borrow check per
//! access on the hottest loop in the engine. The arena also makes `children` an
//! ordered `Vec`, which matters: Python's `dict` is insertion-ordered and
//! `_select_child` breaks ties by taking the FIRST child at the best score, so
//! the order is load-bearing rather than incidental.

use crate::board::Color;
use crate::movegen::Move;
use crate::position::Position;

/// A proven game-theoretic result, from THIS node's side-to-move perspective,
/// with the distance in plies to the mate.
///
/// `Loss(0)` is checkmate on the board: the side to move has already lost. A
/// draw is deliberately NOT represented -- proving a draw needs the fifty-move
/// and repetition state, which is path-dependent, and getting it wrong would
/// claim draws that do not exist. Mates are path-independent and safe.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Proven {
    /// The side to move here forces a mate in this many plies.
    Win(u16),
    /// The side to move here is mated in this many plies against best defence.
    Loss(u16),
}

#[derive(Clone, Debug)]
pub struct Node {
    /// P: the policy's probability for the move that reached here.
    pub prior: f64,
    /// N. Signed because with `vloss_fix` off, virtual loss is still removed by
    /// backing up with sign -1 -- see `pending` for the fixed path instead.
    pub visits: i64,
    /// W, accumulated from this node's OWN side-to-move perspective. With
    /// `vloss_fix` on, this is touched ONLY by real backups: virtual loss no
    /// longer passes through here at all, so it can never corrupt Q.
    pub value_sum: f64,
    /// In-flight simulations, `vloss_fix` only. Lc0's virtual loss: it widens
    /// the PUCT denominator so a second descent in the same batch is
    /// discouraged from re-selecting a child another descent is already
    /// heading into, without moving `value_sum`/`visits` at all. Applied in
    /// `apply_virtual_loss`, undone in `remove_virtual_loss`, read only in
    /// `select_child`'s denominator term. Always 0 when `vloss_fix` is off.
    pub pending: i64,
    /// Ordered, mirroring Python's insertion-ordered dict. Index into the arena.
    pub children: Vec<(Move, usize)>,
    /// Set for checkmate/stalemate/draw; never re-evaluated once set.
    pub terminal_value: Option<f64>,
    /// A proven mate result, once one is known. See `Proven`.
    pub proven: Option<Proven>,
    pub to_move: Color,
}

impl Node {
    fn new(prior: f64, to_move: Color) -> Node {
        Node {
            prior,
            visits: 0,
            value_sum: 0.0,
            pending: 0,
            children: Vec::new(),
            terminal_value: None,
            proven: None,
            to_move,
        }
    }

    #[inline]
    fn expanded(&self) -> bool {
        !self.children.is_empty() || self.terminal_value.is_some()
    }
}

/// What the search asks Python for.
///
/// # Why the action indices are passed in
///
/// The obvious contract is "here are the FENs, give me priors". It was measured
/// and rejected: it forces Python to rebuild a `chess.Board` from each FEN and
/// regenerate its legal moves, which puts `python-chess` back in the hot path
/// and costs a FEN serialise/parse round-trip that the pure-Python search never
/// pays. Measured at 78 ms of a 200 ms callback for a 1600-simulation search.
///
/// So Rust sends the action index for each legal move, in move order, and Python
/// only gathers and softmaxes.
///
/// # Why the softmax stays in Python
///
/// Because it must, to keep bit-identity. `_softmax_over_legal` normalises with
/// `numpy`'s `sum`, which is pairwise rather than sequential. A naive f64 sum in
/// Rust was measured to differ from it on **41% of positions**, by 1 ULP
/// (2.2e-16). That is small enough to look ignorable and is not: a 1-ULP prior
/// difference can flip a PUCT tie, and a flipped tie is a different move.
/// Moving the softmax into Rust means reproducing numpy's summation order, which
/// is a separate piece of work with its own test.
pub trait Evaluator {
    /// `fens[i]` with `actions[i]` (one action index per legal move, in move
    /// order, -1 for a move outside the action space) in; `(priors, values)`
    /// out, with `priors[i]` aligned to that same move order.
    fn evaluate(
        &mut self,
        fens: &[String],
        actions: &[Vec<i32>],
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>), String>;
}

/// Action indices for a move list, via the generated table.
fn actions_for(moves: &[Move]) -> Vec<i32> {
    let table = &*crate::action_space::ACTION_BY_MOVE_KEY;
    moves
        .iter()
        .map(|m| {
            let key = crate::action_space::move_key(
                m.from,
                m.to,
                m.promotion.map(|p| p as u8).unwrap_or(0),
            );
            table[key as usize]
        })
        .collect()
}

pub struct Mcts {
    pub c_puct: f64,
    /// `None` restores the flat constant, which is what `match.py
    /// --a-fixed-cpuct` selects.
    pub c_puct_base: Option<f64>,
    pub c_puct_init: f64,
    pub fpu: f64,
    pub batch: usize,
    pub nodes: Vec<Node>,
    pub evaluations: u64,

    /// Propagate proven mates and prefer the shortest one. Off by default: the
    /// faithful port has no mate distance at all, which is the defect this fixes.
    ///
    /// This IS a behaviour change and it will fail the identity test. That is the
    /// point of the flag.
    pub mate_distance: bool,
    /// Deduplicate the network call when several descents in one batch land on
    /// the SAME leaf. Off by default, because the faithful port must reproduce
    /// the Python's duplicated work.
    ///
    /// This is identity-preserving when done correctly -- see `simulate_batch`.
    pub dedup: bool,
    /// Give virtual loss its own counter (`Node::pending`) instead of routing
    /// it through `value_sum`/`visits`. Off by default, because the faithful
    /// port must reproduce the Python's Q corruption during a batch.
    ///
    /// This is NOT identity-preserving even at fixed simulations: with it on,
    /// mid-batch selection sees different (uncorrupted) Q values, so later
    /// descents in the same batch can choose different children. That is the
    /// fix, not a bug in it -- see `apply_virtual_loss`/`remove_virtual_loss`.
    pub vloss_fix: bool,
    /// Distinct positions actually sent to the evaluator. With `dedup` off this
    /// equals `evaluations`; with it on, the gap IS the duplication.
    pub unique_evaluations: u64,

    // ---- tree reuse ----
    pub reuse: bool,
    /// Simulations inherited from the previous move's tree. `evaluations` counts
    /// only what this search paid for, so nps stays honest; this is the head
    /// start that made it cheaper.
    pub reused: i64,
    /// The move stack the retained tree's root sits at. Copied, because the
    /// caller's board keeps moving.
    root_stack: Vec<Move>,
    /// Whether `nodes[0]` is a retained tree rather than scratch.
    has_tree: bool,
}

impl Mcts {
    pub fn new(c_puct: f64, c_puct_base: Option<f64>, c_puct_init: f64, fpu: f64, batch: usize) -> Mcts {
        Mcts {
            c_puct,
            c_puct_base,
            c_puct_init,
            fpu,
            batch: batch.max(1),
            nodes: Vec::with_capacity(4096),
            evaluations: 0,
            mate_distance: false,
            dedup: false,
            vloss_fix: false,
            unique_evaluations: 0,
            reuse: false,
            reused: 0,
            root_stack: Vec::new(),
            has_tree: false,
        }
    }

    /// Forget the tree. Call between games, never within one.
    pub fn reset(&mut self) {
        self.nodes.clear();
        self.root_stack.clear();
        self.has_tree = false;
        self.reused = 0;
    }

    /// AlphaZero's published schedule. Matches `mcts.py::c_puct_at` exactly,
    /// including the `1.0 +` inside the log.
    #[inline]
    fn c_puct_at(&self, visits: i64) -> f64 {
        match self.c_puct_base {
            None => self.c_puct,
            Some(base) => ((1.0 + visits as f64 + base) / base).ln() + self.c_puct_init,
        }
    }

    /// `mcts.py::_select_child`, expression for expression.
    ///
    /// The hoisting is not an optimisation choice made here, it is what the
    /// Python does, and the arithmetic must associate the same way:
    /// `q + c_sqrt * prior / (1 + visits)` groups as
    /// `q + ((c_sqrt * prior) / (1 + visits))`.
    ///
    /// The comparison is strict `>`, so the FIRST child at the best score wins.
    /// That is why child order is part of the contract.
    fn select_child(&self, node_ix: usize) -> (Move, usize) {
        let node = &self.nodes[node_ix];

        // `effective_n` folds in-flight simulations into N, same as `visits`
        // alone would if virtual loss still ran through `backup`. Without
        // this, a parent sitting at 0 REAL visits (true for every node until
        // its first batch finishes) would score sqrt(0) = 0 for the whole
        // exploration term, and every descent in the batch would pile onto
        // the same first child instead of being spread by the in-flight
        // count below -- silently defeating the point of virtual loss. Only
        // `fpu_q`, which is this node's own Q, stays on REAL visits: that is
        // the one place `vloss_fix` must not touch, or it is not the fix.
        let effective_n = node.visits + node.pending;
        let c_sqrt = self.c_puct_at(effective_n) * (effective_n as f64).sqrt();
        let mut fpu_q = if node.visits != 0 {
            node.value_sum / node.visits as f64 + self.fpu
        } else {
            self.fpu
        };
        if fpu_q < 0.0 {
            fpu_q = 0.0;
        }

        // With mate distance on, proofs override the statistics entirely.
        //
        // A child that is a proven Loss is a win for US, so take the shortest
        // one and stop exploring -- no number of simulations can improve on a
        // proof. A child that is a proven Win loses for us, so it is refuted and
        // worth no further simulations at all, which is where the saving comes
        // from: Lc0 measured ~50% fewer nodes on tactical positions.
        if self.mate_distance {
            let mut shortest_win: Option<(u16, Move, usize)> = None;
            for &(mv, cix) in &node.children {
                if let Some(Proven::Loss(d)) = self.nodes[cix].proven {
                    if shortest_win.is_none_or(|(bd, _, _)| d < bd) {
                        shortest_win = Some((d, mv, cix));
                    }
                }
            }
            if let Some((_, mv, cix)) = shortest_win {
                return (mv, cix);
            }
        }

        let mut best: Option<(Move, usize)> = None;
        let mut best_score = -1e18f64;
        let mut fallback: Option<(Move, usize)> = None;
        for &(mv, child_ix) in &node.children {
            let child = &self.nodes[child_ix];
            if self.mate_distance {
                if let Some(Proven::Win(_)) = child.proven {
                    // Refuted: it mates us. Keep one in reserve in case every
                    // child is refuted, in which case we are lost and must still
                    // return a legal move.
                    if fallback.is_none() {
                        fallback = Some((mv, child_ix));
                    }
                    continue;
                }
            }
            let visits = child.visits;
            let q = if visits != 0 {
                1.0 - child.value_sum / visits as f64
            } else {
                fpu_q
            };
            // `child.pending` is always 0 with `vloss_fix` off, so this is a
            // no-op for the faithful port. With it on, a child another descent
            // in this same batch is already heading into looks less inviting
            // to PUCT's exploration term -- exactly Lc0's denominator-only
            // virtual loss -- while `q` above stays untouched, read from real
            // `value_sum`/`visits` only.
            let denom = (1 + visits + child.pending) as f64;
            let score = q + c_sqrt * child.prior / denom;
            if score > best_score {
                best_score = score;
                best = Some((mv, child_ix));
            }
        }
        best.or(fallback)
            .expect("select_child on a node with no children")
    }

    /// Walk down until an unexpanded node. Returns the path and how many plies
    /// were pushed onto `pos`.
    fn descend(&self, root_ix: usize, pos: &mut Position) -> (Vec<usize>, usize) {
        let mut path = vec![root_ix];
        let mut ix = root_ix;
        let mut undo = 0;
        while self.nodes[ix].expanded() && !self.nodes[ix].children.is_empty() {
            let (mv, child_ix) = self.select_child(ix);
            pos.push(mv);
            undo += 1;
            ix = child_ix;
            path.push(ix);
        }
        (path, undo)
    }

    /// Can this node be proven from its children? `mcts.py` has no equivalent;
    /// this is the MCTS-Solver rule that Lc0 ships in production.
    ///
    /// Children's results are from THEIR side-to-move perspective, so a child
    /// that is a proven Loss is a proven Win for us -- the same `1 - q` flip the
    /// rest of the file lives by.
    ///
    /// * any child a proven Loss  -> we Win, by the SHORTEST such line
    /// * all children proven Wins -> we Lose, by the LONGEST such line
    ///
    /// The min/max asymmetry is the whole point: mate fastest, and be mated as
    /// slowly as possible so a defender prefers the line that gives the opponent
    /// most chance to go wrong.
    fn prove(&mut self, ix: usize) {
        if self.nodes[ix].proven.is_some() || self.nodes[ix].children.is_empty() {
            return;
        }
        let mut best_win: Option<u16> = None;
        let mut worst_loss: Option<u16> = None;
        let mut all_wins = true;
        for &(_, cix) in &self.nodes[ix].children {
            match self.nodes[cix].proven {
                Some(Proven::Loss(d)) => {
                    best_win = Some(best_win.map_or(d, |b: u16| b.min(d)));
                    all_wins = false;
                }
                Some(Proven::Win(d)) => {
                    worst_loss = Some(worst_loss.map_or(d, |b: u16| b.max(d)));
                }
                None => all_wins = false,
            }
        }
        if let Some(d) = best_win {
            self.nodes[ix].proven = Some(Proven::Win(d + 1));
        } else if all_wins {
            if let Some(d) = worst_loss {
                self.nodes[ix].proven = Some(Proven::Loss(d + 1));
            }
        }
    }

    /// `mcts.py::_backup`. Flips at every level; `sign` -1 removes a virtual
    /// loss by subtracting exactly what was added.
    fn backup(&mut self, path: &[usize], value: f64, sign: i64) {
        let mut value = value;
        for &ix in path.iter().rev() {
            let n = &mut self.nodes[ix];
            n.visits += sign;
            n.value_sum += sign as f64 * value;
            value = 1.0 - value;
            // Proving runs on the real backup only. Doing it during the virtual
            // loss or its removal would prove nodes from provisional state, and
            // a proof is not something to take back.
            if self.mate_distance && sign > 0 {
                self.prove(ix);
            }
        }
    }

    /// `vloss_fix` virtual loss: mark every node on `path` as having one more
    /// in-flight simulation. Touches `pending` only -- never `visits` or
    /// `value_sum` -- so it cannot corrupt Q the way `backup(path, 1.0, 1)`
    /// does for the faithful port. See `Node::pending` and `select_child`.
    fn apply_virtual_loss(&mut self, path: &[usize]) {
        for &ix in path {
            self.nodes[ix].pending += 1;
        }
    }

    /// Undo `apply_virtual_loss` for the same path, once the real value is
    /// ready to back up for real.
    fn remove_virtual_loss(&mut self, path: &[usize]) {
        for &ix in path {
            self.nodes[ix].pending -= 1;
        }
    }

    fn expand_from(&mut self, node_ix: usize, priors: &[f64], moves: &[Move]) {
        let to_move = self.nodes[node_ix].to_move;
        let mut children = Vec::with_capacity(moves.len());
        for (i, &mv) in moves.iter().enumerate() {
            let child = Node::new(priors[i], to_move.other());
            self.nodes.push(child);
            children.push((mv, self.nodes.len() - 1));
        }
        self.nodes[node_ix].children = children;
    }

    /// One batch of simulations. `mcts.py::_simulate_batch`.
    ///
    /// # Leaf deduplication, and why it does not change the tree
    ///
    /// Two descents in one batch can reach the same unexpanded leaf: nothing
    /// expands it until after the network call, so the second walk stops there
    /// too. The Python sends both to the network. Measured on a real position,
    /// batch 1024 was **98.6% duplicated work** -- 37 distinct positions per
    /// second out of 2,656 raw.
    ///
    /// The fix is to dedupe the **evaluation** and NOT the **backup**. k paths to
    /// one leaf get one network row and still k backups, so the tree ends in
    /// exactly the state it would have: k visits, k copies of the value. The only
    /// thing that changes is how many rows the network was asked for.
    ///
    /// # The ordering trap
    ///
    /// Backups must still run in DESCENT order. f64 addition is not associative,
    /// so grouping the backups by leaf -- the obvious way to write this -- would
    /// change `value_sum` in the last bits and break the identity test for a
    /// reason that has nothing to do with the idea being wrong. So `pending`
    /// stays one entry per path, in order, and only the evaluation list is
    /// deduplicated.
    fn simulate_batch<E: Evaluator>(
        &mut self,
        root_ix: usize,
        pos: &mut Position,
        batch: usize,
        ev: &mut E,
    ) -> Result<(), String> {
        // One entry per PATH, in descent order. `slot` indexes `distinct`.
        struct Pending {
            path: Vec<usize>,
            slot: usize,
        }
        let mut pending: Vec<Pending> = Vec::with_capacity(batch);
        // One entry per distinct leaf actually sent to the evaluator.
        let mut distinct: Vec<(usize, String, Vec<Move>)> = Vec::with_capacity(batch);

        for _ in 0..batch {
            let (path, undo) = self.descend(root_ix, pos);
            let leaf_ix = *path.last().unwrap();

            if let Some(tv) = self.nodes[leaf_ix].terminal_value {
                self.backup(&path, tv, 1);
                for _ in 0..undo {
                    pos.pop();
                }
                continue;
            }

            if let Some(tv) = pos.terminal_value() {
                self.nodes[leaf_ix].terminal_value = Some(tv);
                // A terminal value of 0.0 with the side to move in check is
                // checkmate: already lost, distance zero. A 0.5 is a draw and is
                // deliberately NOT proven -- see `Proven`.
                if self.mate_distance && tv == 0.0 {
                    self.nodes[leaf_ix].proven = Some(Proven::Loss(0));
                }
                self.backup(&path, tv, 1);
                for _ in 0..undo {
                    pos.pop();
                }
                continue;
            }

            // Has this leaf already been queued in this batch?
            let existing = if self.dedup {
                distinct.iter().position(|(ix, _, _)| *ix == leaf_ix)
            } else {
                None
            };
            let slot = match existing {
                Some(s) => s,
                None => {
                    let moves = crate::movegen::generate_legal_moves(&pos.board);
                    distinct.push((leaf_ix, pos.board.to_fen(), moves));
                    distinct.len() - 1
                }
            };
            pending.push(Pending { path: path.clone(), slot });
            // Faithful port: virtual loss IS a real backup with sign 1,
            // reverted below with sign -1 -- that is the defect (it passes
            // through `value_sum`). Fixed: it only marks `pending`, which
            // `select_child` reads and real backup never touches.
            if self.vloss_fix {
                self.apply_virtual_loss(&path);
            } else {
                self.backup(&path, 1.0, 1); // virtual loss, removed below
            }
            for _ in 0..undo {
                pos.pop();
            }
        }

        if pending.is_empty() {
            return Ok(());
        }

        let fens: Vec<String> = distinct.iter().map(|(_, f, _)| f.clone()).collect();
        let actions: Vec<Vec<i32>> = distinct.iter().map(|(_, _, mv)| actions_for(mv)).collect();
        let (priors, values) = ev.evaluate(&fens, &actions)?;
        if priors.len() != distinct.len() || values.len() != distinct.len() {
            return Err(format!(
                "evaluator returned {} priors and {} values for {} positions",
                priors.len(),
                values.len(),
                distinct.len()
            ));
        }
        // `evaluations` counts what the Python would have counted, so nps stays
        // comparable across the flag; `unique_evaluations` is the honest one.
        self.evaluations += pending.len() as u64;
        self.unique_evaluations += distinct.len() as u64;

        for p in &pending {
            let leaf_ix = *p.path.last().unwrap();
            // Another path in this same batch may already have expanded it.
            if self.nodes[leaf_ix].children.is_empty() {
                let (_, _, moves) = &distinct[p.slot];
                if priors[p.slot].len() != moves.len() {
                    return Err(format!(
                        "evaluator returned {} priors for a position with {} legal moves",
                        priors[p.slot].len(),
                        moves.len()
                    ));
                }
                self.expand_from(leaf_ix, &priors[p.slot], &moves.clone());
            }
            if self.vloss_fix {
                self.remove_virtual_loss(&p.path);
            } else {
                self.backup(&p.path, 1.0, -1); // remove the virtual loss
            }
            self.backup(&p.path, values[p.slot], 1); // add the real result
        }

        Ok(())
    }

    /// Copy the subtree rooted at `ix` into a fresh arena, remapping indices.
    ///
    /// Without this, rerooting would leave the whole discarded tree in the arena
    /// and it would grow for the entire game -- Python gets this for free from
    /// refcounting, Rust does not. Cost is O(retained subtree), which is exactly
    /// the thing being kept, and far below the cost of rebuilding it.
    ///
    /// Recursion is on tree DEPTH, not size, so a large subtree is fine.
    fn extract_subtree(old: &[Node], ix: usize, out: &mut Vec<Node>) -> usize {
        let new_ix = out.len();
        out.push(old[ix].clone());
        let mut children = Vec::with_capacity(old[ix].children.len());
        for &(mv, child_ix) in &old[ix].children {
            let n = Self::extract_subtree(old, child_ix, out);
            children.push((mv, n));
        }
        out[new_ix].children = children;
        new_ix
    }

    /// The stored node for this position, or `None` to start fresh.
    ///
    /// `mcts.py::_reroot`, condition for condition. It DECLINES rather than
    /// guesses: a new game, a jumped-to position, an analysis GUI walking
    /// backwards, or a board built without history all land here and all get a
    /// fresh root, which is merely the old behaviour.
    ///
    /// Keyed on the move stack, not the FEN. Comparing FENs would be wrong:
    /// two lines can transpose into the same position with different repetition
    /// histories, and reusing across a transposition would import a subtree whose
    /// threefold evaluations were computed against the wrong history.
    fn reroot(&mut self, pos: &Position) -> Option<usize> {
        if !self.has_tree || self.nodes.is_empty() {
            return None;
        }
        let stack = pos.moves();
        let played = stack.len().checked_sub(self.root_stack.len())?;
        // Two plies is the normal case: our move and their reply. One covers
        // being asked to move twice in a row (analysis). More than that and we
        // have missed part of the game.
        if !(1..=2).contains(&played) {
            return None;
        }
        if stack[..self.root_stack.len()] != self.root_stack[..] {
            return None;
        }

        let mut ix = 0usize; // the retained tree's root is always index 0
        for mv in &stack[self.root_stack.len()..] {
            let child = self.nodes[ix]
                .children
                .iter()
                .find(|(m, _)| m == mv)
                .map(|&(_, c)| c)?;
            // An unexpanded child has no subtree worth keeping, and `search`
            // would have to expand it anyway.
            if !self.nodes[child].expanded() {
                return None;
            }
            ix = child;
        }

        // It is a root now, so nothing scores it by its prior any more.
        let mut fresh = Vec::with_capacity(self.nodes.len());
        Self::extract_subtree(&self.nodes, ix, &mut fresh);
        fresh[0].prior = 1.0;
        self.nodes = fresh;
        Some(0)
    }

    /// Run `simulations` simulations from `pos`. Returns the root index.
    ///
    /// `pos` must carry the game's history, because repetition depends on the
    /// moves before the root. A `Position` built from a bare FEN has an empty
    /// stack and will disagree with Python about threefold draws.
    pub fn search<E: Evaluator>(
        &mut self,
        pos: &mut Position,
        simulations: u32,
        ev: &mut E,
    ) -> Result<usize, String> {
        self.evaluations = 0;
        self.unique_evaluations = 0;

        let reused_root = if self.reuse { self.reroot(pos) } else { None };
        self.reused = match reused_root {
            Some(ix) => self.nodes[ix].visits,
            None => 0,
        };
        if reused_root.is_none() {
            self.nodes.clear();
            self.nodes.push(Node::new(1.0, pos.board.turn));
        }
        let root_ix = 0usize;

        // Root expansion. A reused root arrives already expanded and already
        // carrying visits, so expanding it again would overwrite its children
        // and destroy the whole point -- which is why this is conditional and
        // why it is not optional. Note the Python discards the returned value
        // but DOES count the evaluation, so the node budget is off by one from
        // the simulation count. Reproduced.
        if !self.nodes[root_ix].expanded() {
            if let Some(tv) = pos.terminal_value() {
                self.nodes[root_ix].terminal_value = Some(tv);
            } else {
                let moves = crate::movegen::generate_legal_moves(&pos.board);
                let fens = vec![pos.board.to_fen()];
                let actions = vec![actions_for(&moves)];
                let (priors, _values) = ev.evaluate(&fens, &actions)?;
                if priors[0].len() != moves.len() {
                    return Err("root prior length mismatch".to_string());
                }
                self.expand_from(root_ix, &priors[0], &moves);
                self.evaluations += 1;
            }
        }

        // Dirichlet noise is not ported: `dirichlet_weight` defaults to 0.0 and
        // is only wanted for self-play data generation, never for play. If it is
        // ever needed, it must use the same RNG stream as numpy's to preserve
        // the identity test, which is a real cost worth knowing about up front.

        let mut done = 0u32;
        while done < simulations {
            let n = std::cmp::min(self.batch as u32, simulations - done);
            self.simulate_batch(root_ix, pos, n as usize, ev)?;
            done += n;
        }

        if self.reuse {
            self.root_stack = pos.moves();
            self.has_tree = true;
        }

        Ok(root_ix)
    }

    /// The root's proven result, if any: `(true, plies)` for a forced win,
    /// `(false, plies)` for a forced loss. Exposed so the mate suite can check a
    /// claimed proof against an independent solver rather than trusting it.
    pub fn root_proven(&self, root_ix: usize) -> Option<(bool, u16)> {
        match self.nodes[root_ix].proven {
            Some(Proven::Win(d)) => Some((true, d)),
            Some(Proven::Loss(d)) => Some((false, d)),
            None => None,
        }
    }

    /// Root children as `(uci, visits)`, in child order.
    pub fn root_visits(&self, root_ix: usize) -> Vec<(String, i64)> {
        self.nodes[root_ix]
            .children
            .iter()
            .map(|&(mv, ix)| (mv.uci(), self.nodes[ix].visits))
            .collect()
    }

    /// The most-visited root move, ties going to the FIRST such child.
    ///
    /// `mcts.py::play` uses `max(visits.items(), key=lambda kv: kv[1])`, and
    /// Python's `max` returns the first maximal element, so a strict `>` here
    /// matches. Visit count rather than average value: a child can luck into a
    /// high average from two visits, but it only accumulates visits by
    /// surviving PUCT repeatedly.
    pub fn best_move(&self, root_ix: usize) -> Option<String> {
        // A proof beats a visit count. Shortest mate first; ties to the first
        // child, matching the visit-count rule.
        if self.mate_distance {
            let mut shortest: Option<(u16, &Move)> = None;
            for (mv, ix) in &self.nodes[root_ix].children {
                if let Some(Proven::Loss(d)) = self.nodes[*ix].proven {
                    if shortest.is_none_or(|(bd, _)| d < bd) {
                        shortest = Some((d, mv));
                    }
                }
            }
            if let Some((_, mv)) = shortest {
                return Some(mv.uci());
            }
        }
        let mut best: Option<(&Move, i64)> = None;
        for (mv, ix) in &self.nodes[root_ix].children {
            let v = self.nodes[*ix].visits;
            match best {
                Some((_, bv)) if v <= bv => {}
                _ => best = Some((mv, v)),
            }
        }
        best.map(|(mv, _)| mv.uci())
    }
}

#[cfg(test)]
mod vloss_fix_tests {
    //! `vloss_fix` is otherwise only proven correct by construction (it never
    //! calls `backup` with the virtual-loss sign, so it structurally cannot
    //! touch `value_sum`). These tests catch the two ways that could still be
    //! wrong in practice: the flag silently doing nothing (a wiring bug), and
    //! virtual loss leaking (an unpaired apply/remove).
    use super::*;
    use crate::board::Board;

    const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    /// Returns a fixed value and uniform priors for every position, so any
    /// drift in `value_sum` can only have come from virtual loss, never from
    /// the (nonexistent) skill of the evaluator.
    struct FixedEval {
        value: f64,
    }

    impl Evaluator for FixedEval {
        fn evaluate(
            &mut self,
            fens: &[String],
            actions: &[Vec<i32>],
        ) -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
            let priors = actions
                .iter()
                .map(|a| {
                    let n = a.len().max(1) as f64;
                    vec![1.0 / n; a.len()]
                })
                .collect();
            let values = vec![self.value; fens.len()];
            Ok((priors, values))
        }
    }

    fn run(vloss_fix: bool, sims: u32, batch: usize, value: f64) -> Mcts {
        let mut mcts = Mcts::new(2.0, Some(19652.0), 1.25, -0.2, batch);
        mcts.vloss_fix = vloss_fix;
        let mut pos = Position::new(Board::from_fen(STARTPOS).unwrap());
        let mut ev = FixedEval { value };
        mcts.search(&mut pos, sims, &mut ev).unwrap();
        mcts
    }

    #[test]
    fn off_never_leaves_pending_nonzero_because_it_never_sets_it() {
        let mcts = run(false, 256, 32, 0.5);
        assert!(mcts.nodes.iter().all(|n| n.pending == 0));
    }

    #[test]
    fn on_removes_every_virtual_loss_it_applies() {
        // A batch this size against a branching root guarantees several
        // descents share ancestors before any evaluation returns, which is
        // exactly the window where a leak would show up.
        let mcts = run(true, 512, 64, 0.5);
        assert!(
            mcts.nodes.iter().all(|n| n.pending == 0),
            "virtual loss left residue: apply/remove is not paired"
        );
    }

    #[test]
    fn on_never_lets_value_sum_carry_anything_but_real_backups() {
        // 0.5 is its own complement (1.0 - 0.5 == 0.5), so the alternating
        // sign-flip in `backup` contributes exactly 0.5 per visit at every
        // depth. If ANY virtual loss (which always adds/removes 1.0, not the
        // evaluator's value) had leaked into `value_sum`, this exact equality
        // would fail -- there is no rounding budget for it to hide in, since
        // 0.5 and integers are exactly representable.
        let mcts = run(true, 512, 64, 0.5);
        for n in &mcts.nodes {
            if n.visits > 0 {
                let expected = n.visits as f64 * 0.5;
                assert_eq!(
                    n.value_sum, expected,
                    "value_sum {} != visits*0.5 = {} (visits={}): virtual loss touched it",
                    n.value_sum, expected, n.visits
                );
            }
        }
    }

    #[test]
    fn fix_is_not_a_silent_no_op() {
        // Same position, same evaluator, same seed-free deterministic search,
        // batch > 1 so mid-batch selection is actually exercised. If wiring
        // the flag through PyMcts/RustMCTS/simulate_batch had a bug that left
        // it inert, this would be the one thing that could not tell the
        // difference -- so assert there IS one.
        let off = run(false, 800, 64, 0.5);
        let on = run(true, 800, 64, 0.5);
        assert_ne!(
            off.root_visits(0),
            on.root_visits(0),
            "vloss_fix changed nothing: it is wired to a no-op"
        );
    }

    #[test]
    fn default_construction_is_off_and_unaffected() {
        let mcts = Mcts::new(2.0, Some(19652.0), 1.25, -0.2, 32);
        assert!(!mcts.vloss_fix);
    }
}
