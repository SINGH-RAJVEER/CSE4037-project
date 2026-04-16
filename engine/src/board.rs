use ndarray::Array4;
use shakmaty::{Chess, Color, EnPassantMode, Move, Position, Role};

const QUEEN_DIRS: [(i32, i32); 8] = [
    (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1),
];
const KNIGHT_OFFSETS: [(i32, i32); 8] = [
    (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1),
];

fn piece_plane(role: Role, color: Color) -> usize {
    let base = match role {
        Role::Pawn   => 0, Role::Knight => 1, Role::Bishop => 2,
        Role::Rook   => 3, Role::Queen  => 4, Role::King   => 5,
    };
    if color == Color::White { base } else { base + 6 }
}

// Planes 0-5  : white  {P N B R Q K}
// Planes 6-11 : black  {P N B R Q K}
// Plane  12   : side-to-move (all 1s = white)
// Plane  13   : en-passant target square
pub fn board_to_tensor(pos: &Chess) -> Array4<f32> {
    let mut t = Array4::<f32>::zeros((1, 14, 8, 8));
    for (sq, piece) in pos.board().clone() {
        let plane = piece_plane(piece.role, piece.color);
        t[[0, plane, sq.rank() as usize, sq.file() as usize]] = 1.0;
    }
    if pos.turn() == Color::White {
        for rank in 0..8usize {
            for file in 0..8usize {
                t[[0, 12, rank, file]] = 1.0;
            }
        }
    }
    if let Some(ep) = pos.ep_square(EnPassantMode::Legal) {
        t[[0, 13, ep.rank() as usize, ep.file() as usize]] = 1.0;
    }
    t
}

// index = plane * 64 + from_rank * 8 + from_file
// Planes  0-55: queen moves  dir*7 + (dist-1),  dirs: N NE E SE S SW W NW
// Planes 56-63: knight moves (8 offsets)
// Planes 64-72: underpromotions  piece*3 + (df+1),  pieces={N,B,R}
pub fn move_to_policy_index(m: &Move) -> Option<usize> {
    let (from_sq, to_sq, promo) = match m {
        Move::Normal    { from, to, promotion, .. } => (*from, *to, *promotion),
        Move::EnPassant { from, to }                => (*from, *to, None),
        Move::Castle    { king, rook }              => {
            let fr = king.rank() as i32;
            let ff = king.file() as i32;
            let df = if rook.file() as i32 > ff { 2 } else { -2 };
            let dir_i = if df > 0 { 2usize } else { 6usize };
            let plane = dir_i * 7 + 1;
            return Some(plane * 64 + fr as usize * 8 + ff as usize);
        }
        Move::Put { .. } => return None,
    };

    let fr = from_sq.rank() as i32;
    let ff = from_sq.file() as i32;
    let dr = to_sq.rank() as i32 - fr;
    let df = to_sq.file() as i32 - ff;
    let flat_from = fr as usize * 8 + ff as usize;

    if let Some(p) = promo {
        let promo_idx = match p {
            Role::Knight => Some(0usize),
            Role::Bishop => Some(1),
            Role::Rook   => Some(2),
            _            => None,
        };
        if let Some(pi) = promo_idx {
            let plane = 64 + pi * 3 + (df + 1) as usize;
            return Some(plane * 64 + flat_from);
        }
    }

    for (i, &(nr, nf)) in KNIGHT_OFFSETS.iter().enumerate() {
        if dr == nr && df == nf {
            return Some((56 + i) * 64 + flat_from);
        }
    }

    for (di, &(dir_r, dir_f)) in QUEEN_DIRS.iter().enumerate() {
        for dist in 1i32..=7 {
            if dr == dir_r * dist && df == dir_f * dist {
                return Some((di * 7 + (dist - 1) as usize) * 64 + flat_from);
            }
        }
    }
    None
}
