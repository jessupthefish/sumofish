//! The `Panel` trait, and the contract that makes adding one cheap.
//!
//! In the Python, adding a panel cost eight edit sites across five files
//! (measured from commit `772fc12`), and nothing enumerated panels, so nothing
//! could be added by declaration. The cause was that a panel was a free function
//! with an ad-hoc positional signature, its size was pushed in as `width`/`height`
//! ints by a caller that had computed them by hand, and `draw()` was a second
//! hardcoded list of the same panels.
//!
//! Here a panel declares what it needs and the solver decides. `render` is pure:
//! no I/O, no clock, no mutation. That is enforced structurally -- `sf-panels`
//! depends on no crate that can do I/O -- and the clock is closed off by handing
//! `now` in through [`Cx`] rather than letting a panel call `Instant::now`, which
//! is what makes a ticking clock snapshot-testable.

use ratatui::buffer::Buffer;
use ratatui::crossterm::event::KeyEvent;
use ratatui::layout::Rect;
use sf_model::AppState;
use sf_theme::Track;
use std::time::Instant;

/// Stable identity. Used for config, drop order, snapshot filenames and the
/// per-panel view state slot. Never derived from the title, so retitling a panel
/// cannot silently reset its state or rename its snapshots.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PanelId(pub &'static str);

impl std::fmt::Display for PanelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

/// Where a panel lives. Panels name a region; compositions arrange regions. That
/// indirection is what lets a tier rearrange the screen without any panel
/// knowing.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum RegionId {
    /// The board column. Special: it hosts the picture.
    Board,
    /// The primary right-hand column.
    Rail,
    /// The second column, which only the `Wide` tier has.
    RailB,
    /// A full-width strip under everything.
    Band,
    /// The whole screen, one panel at a time. `Micro` only.
    Full,
}

impl RegionId {
    pub const ALL: [RegionId; 5] =
        [RegionId::Board, RegionId::Rail, RegionId::RailB, RegionId::Band, RegionId::Full];
    pub const fn name(self) -> &'static str {
        match self {
            RegionId::Board => "board",
            RegionId::Rail => "rail",
            RegionId::RailB => "rail-b",
            RegionId::Band => "band",
            RegionId::Full => "full",
        }
    }
}

/// Whether a panel is about the machine, the focused bot, or every bot.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scope {
    Global,
    FocusedBot,
    AllBots,
}

/// Drop order within a region when there is not enough room. `Pinned` is never
/// dropped, and the composition's floor guarantees the pinned set always fits --
/// which is what makes the solver's "nothing left to drop" branch unreachable
/// rather than a runtime surprise.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Weight {
    /// Dropped first.
    Low,
    Normal,
    High,
    Pinned,
}

/// How much detail a variant shows. Declared most-detailed first.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum VariantId {
    Full,
    Compact,
    Line,
    Glyph,
}

impl VariantId {
    pub const fn name(self) -> &'static str {
        match self {
            VariantId::Full => "full",
            VariantId::Compact => "compact",
            VariantId::Line => "line",
            VariantId::Glyph => "glyph",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Size {
    pub w: u16,
    pub h: u16,
}

impl Size {
    pub const fn new(w: u16, h: u16) -> Self {
        Size { w, h }
    }
    pub fn fits_in(self, r: Rect) -> bool {
        r.width >= self.w && r.height >= self.h
    }
}

/// One level of detail a panel can render at.
#[derive(Clone, Copy, Debug)]
pub struct Variant {
    pub id: VariantId,
    /// Hard floor. Below this the variant is not offered at all, so a panel is
    /// never handed a rect it cannot draw in. This is what replaces the Python's
    /// nine separate open-coded `width - 4` / `height - 8` expressions, one of
    /// which shipped the same off-by-one twice.
    pub min: Size,
    /// Above this, extra space is wasted on this panel; give it to a sibling.
    pub pref: Size,
    /// Share of leftover space along the region's main axis. Zero means fixed.
    pub grow: u16,
}

impl Variant {
    pub const fn new(id: VariantId, min: Size, pref: Size, grow: u16) -> Self {
        Variant { id, min, pref, grow }
    }
    /// A panel whose height never varies.
    pub const fn fixed(id: VariantId, w: u16, h: u16) -> Self {
        Variant { id, min: Size::new(w, h), pref: Size::new(w, h), grow: 0 }
    }
}

/// Everything about a panel that is known at compile time.
#[derive(Clone, Copy, Debug)]
pub struct PanelSpec {
    pub id: PanelId,
    pub title: &'static str,
    pub region: RegionId,
    pub scope: Scope,
    pub weight: Weight,
    /// Most detailed first, `min` strictly decreasing on both axes. Asserted by a
    /// test over the registry, so a badly ordered list cannot ship.
    pub variants: &'static [Variant],
}

/// Does this panel have anything to say right now?
///
/// New, and load-bearing: the Python shows "no training run active" in a
/// six-row box forever, and a permanently red dot for a unit that was deleted.
/// A panel that reports `Hidden` gives its rows to a sibling that has news.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Relevance {
    /// Nothing to say. Drop it.
    Hidden,
    /// Has data but nothing is happening; first to be demoted or dropped.
    Idle,
    Normal,
    /// Something is wrong or interesting. Resists dropping.
    Urgent,
}

/// A request for a raster image inside `area`, *declared* rather than performed.
/// Keeping the demand pure is what makes the picture's geometry a function of
/// (state, area) that a test can assert without a terminal -- the Python
/// recomputed it in three places that could not see each other.
#[derive(Clone, Debug)]
pub struct PictureRequest {
    /// The cells the picture will occupy.
    pub cells: Rect,
    /// FEN of the position to draw.
    pub fen: String,
    /// Draw from Black's side.
    pub flip: bool,
    /// The last move, for the highlight.
    pub last: Option<String>,
    /// The square of a king in check, for the ring.
    pub check: Option<String>,
}

/// A key press turned into data. A panel never mutates anything; the app applies
/// the action. That is what keeps `render` and `on_key` both taking `&self`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
    Quit,
    Redraw,
    FocusNextBot,
    FocusPrevBot,
    FocusBot(usize),
    ScrollUp(PanelId),
    ScrollDown(PanelId),
    ToggleExpand(PanelId),
    ToggleOverlay(sf_model::state::Overlay),
    TogglePause,
}

/// Everything a `render` is allowed to see. There is no other channel: no
/// globals, no I/O, no clock.
pub struct Cx<'a> {
    pub state: &'a AppState,
    /// Monotonic now, for ages and staleness. Passed in rather than read, which
    /// is what makes a frame byte-reproducible in a snapshot test. The Python
    /// called `time.time()` inside `live_clocks` and `time.strftime` inside
    /// `tape_panel`, which makes both unsnapshottable.
    pub now: Instant,
    /// Wall clock, for displaying times of day.
    pub wall: jiff::Timestamp,
    /// Which bot this render is about. `None` for `Scope::Global` panels.
    pub bot: Option<&'a sf_model::BotId>,
    /// Frame counter, for anything that animates deterministically.
    pub frame: u64,
    /// The picture's geometry, when one is on screen. Handed to the board panel
    /// so it can draw the player blocks around it, and it is the *same* value the
    /// emitter uses -- one source of truth replacing the Python's three.
    pub picture: Option<PictureGeom>,
}

impl Cx<'_> {
    /// The bot this render is about, resolved.
    pub fn bot_state(&self) -> Option<&sf_model::BotState> {
        match self.bot {
            Some(id) => self.state.bots.get(id),
            None => self.state.focused(),
        }
    }
    pub fn game(&self) -> Option<&sf_model::GameState> {
        self.bot_state().and_then(|b| b.game())
    }
    /// Convert a model track into a theme track. The two are separate types
    /// because `sf-model` must not depend on `sf-theme`; this is the one seam.
    pub fn track(t: sf_model::Track) -> Track {
        match t {
            sf_model::Track::Live => Track::Live,
            sf_model::Track::Coast => Track::Coast,
            sf_model::Track::Lost => Track::Lost,
        }
    }
}

/// Where the picture goes and how big it is, in both cells and pixels. Computed
/// once, by the solver, and read by the board panel and the emitter alike.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PictureGeom {
    pub cells: Rect,
    /// Edge length in pixels, always a multiple of 26.
    pub px: u32,
    pub cell_w: u16,
    pub cell_h: u16,
}

/// `chess.svg.board` is a 390-unit square: eight 45-unit squares plus a 15-unit
/// margin either side. A square edge therefore lands on an exact pixel only at a
/// multiple of 390/15 = 26; anywhere else the rasteriser blends the two square
/// colours across the boundary and the board grows a hairline grid it does not
/// have. Measured through both rsvg and resvg in the M0 probe: eight stray pixels
/// per scanline at 1152, zero at 1144.
pub const GRID_PX: u32 = 26;

pub fn snap26(px: u32) -> u32 {
    px.saturating_sub(px % GRID_PX).max(GRID_PX)
}

/// The interface every panel implements.
pub trait Panel: Send + Sync + 'static {
    fn spec(&self) -> &'static PanelSpec;

    /// Pure. Default is "always worth showing".
    fn relevance(&self, _cx: &Cx<'_>) -> Relevance {
        Relevance::Normal
    }

    /// Pure. No I/O, no clock, no mutation. Called at most once per frame.
    fn render(&self, variant: VariantId, area: Rect, buf: &mut Buffer, cx: &Cx<'_>);

    /// Declare a demand for a raster image. Only the board panel implements it.
    fn picture(&self, _variant: VariantId, _area: Rect, _cx: &Cx<'_>) -> Option<PictureRequest> {
        None
    }

    fn on_key(&self, _key: KeyEvent, _cx: &Cx<'_>) -> Option<Action> {
        None
    }
}

/// Declare the panel registry. Expands to the `mod` declarations *and* the
/// slice, so adding a panel is one new file plus one identifier here.
///
/// Distributed slices (`inventory`, `linkme`) would get this to zero lines and
/// are deliberately rejected: their order is link-order dependent, this design's
/// safety argument rests on determinism, and the list order here *is* the
/// drop-order tie-break. That one line is a decision, not ceremony.
#[macro_export]
macro_rules! panels {
    ($($name:ident),* $(,)?) => {
        $( pub mod $name; )*
        /// Every panel, in declaration order. The order is the tie-break for
        /// which of two equal-weight panels is dropped first.
        pub static ALL: &[&dyn $crate::Panel] = &[ $( &$name::PANEL ),* ];
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snap26_lands_on_the_grid_and_never_returns_zero() {
        assert_eq!(snap26(1152), 1144);
        assert_eq!(snap26(1144), 1144);
        assert_eq!(snap26(1180), 1170);
        assert_eq!(snap26(0), GRID_PX, "a degenerate size must not produce a 0px image");
        assert_eq!(snap26(25), GRID_PX);
        for px in 0..4000u32 {
            assert_eq!(snap26(px) % GRID_PX, 0, "{px} snapped off-grid");
            assert!(snap26(px) >= GRID_PX);
        }
    }

    #[test]
    fn drop_order_is_low_first() {
        assert!(Weight::Low < Weight::Normal);
        assert!(Weight::Normal < Weight::High);
        assert!(Weight::High < Weight::Pinned);
    }

    #[test]
    fn relevance_orders_hidden_first_and_urgent_last() {
        assert!(Relevance::Hidden < Relevance::Idle);
        assert!(Relevance::Idle < Relevance::Normal);
        assert!(Relevance::Normal < Relevance::Urgent);
    }
}
