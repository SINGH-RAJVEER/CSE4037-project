use shakmaty::{Chess, Color, Position, Role};

pub fn piece_value(role: Role) -> i32 {
    match role {
        Role::Pawn   => 100, Role::Knight => 320, Role::Bishop => 330,
        Role::Rook   => 500, Role::Queen  => 900, Role::King   => 20_000,
    }
}

pub fn evaluate(pos: &Chess) -> i32 {
    let mut score = 0;
    for (_sq, piece) in pos.board().clone() {
        let v = piece_value(piece.role);
        if piece.color == Color::White { score += v; } else { score -= v; }
    }
    score
}

pub fn flip_color(c: Color) -> Color {
    if c == Color::White { Color::Black } else { Color::White }
}
