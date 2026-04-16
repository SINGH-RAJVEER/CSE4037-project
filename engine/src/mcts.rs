use std::collections::HashMap;
use std::time::{Duration, Instant};

use ort::{session::Session, value::Tensor};
use shakmaty::{Chess, Move, Outcome, Position};

use crate::board::{board_to_tensor, move_to_policy_index};
use crate::eval::flip_color;

const C_PUCT: f32 = 1.5;
const NEURAL_TIME: u64 = 3000;
const NEURAL_SIMS: u32 = 800;

struct NeuralNode {
    pos: Chess,
    mv: Option<Move>,
    parent: Option<usize>,
    children: Vec<usize>,
    visits: u32,
    value_sum: f32,
    prior: f32,
    is_expanded: bool,
}

impl NeuralNode {
    fn new(pos: Chess, mv: Option<Move>, parent: Option<usize>, prior: f32) -> Self {
        NeuralNode {
            pos, mv, parent,
            children: vec![],
            visits: 0,
            value_sum: 0.0,
            prior,
            is_expanded: false,
        }
    }
    fn q(&self) -> f32 {
        if self.visits == 0 { 0.0 } else { self.value_sum / self.visits as f32 }
    }
    fn puct(&self, pv: u32) -> f32 {
        self.q() + C_PUCT * self.prior * (pv as f32).sqrt() / (1.0 + self.visits as f32)
    }
}

fn neural_eval(session: &mut Session, pos: &Chess) -> (HashMap<usize, f32>, f32) {
    let flat: Vec<f32> = board_to_tensor(pos).into_raw_vec_and_offset().0;
    let input = Tensor::<f32>::from_array(([1usize, 14, 8, 8], flat)).expect("board tensor");
    let outputs = session.run(ort::inputs!["board" => input]).expect("ort inference");
    let (_, policy_data) = outputs["policy"].try_extract_tensor::<f32>().expect("policy");
    let (_, value_data)  = outputs["value"].try_extract_tensor::<f32>().expect("value");
    let value = value_data[0];

    let legals: Vec<Move> = pos.legal_moves().into_iter().collect();
    let legal_idx: Vec<(usize, f32)> = legals.iter()
        .filter_map(|m| move_to_policy_index(m).map(|i| (i, policy_data[i])))
        .collect();
    if legal_idx.is_empty() { return (HashMap::new(), value); }

    let max_l = legal_idx.iter().map(|&(_, l)| l).fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    let exps: Vec<f32> = legal_idx.iter()
        .map(|&(_, l)| { let e = (l - max_l).exp(); sum += e; e })
        .collect();
    let policy = legal_idx.iter().zip(&exps).map(|(&(i, _), &e)| (i, e / sum)).collect();
    (policy, value)
}

fn neural_expand(arena: &mut Vec<NeuralNode>, idx: usize, session: &mut Session) -> f32 {
    let (policy, value) = neural_eval(session, &arena[idx].pos.clone());
    arena[idx].is_expanded = true;
    for m in arena[idx].pos.legal_moves().into_iter() {
        if let Some(pidx) = move_to_policy_index(&m) {
            let prior = *policy.get(&pidx).unwrap_or(&1e-3);
            let mut cp = arena[idx].pos.clone();
            cp.play_unchecked(&m);
            let ci = arena.len();
            arena[idx].children.push(ci);
            arena.push(NeuralNode::new(cp, Some(m), Some(idx), prior));
        }
    }
    value
}

fn neural_select(arena: &[NeuralNode]) -> usize {
    let mut idx = 0;
    loop {
        let node = &arena[idx];
        if !node.is_expanded || node.pos.is_game_over() || node.children.is_empty() {
            return idx;
        }
        let pv = node.visits;
        idx = *node.children.iter()
            .max_by(|&&a, &&b| arena[a].puct(pv).partial_cmp(&arena[b].puct(pv)).unwrap())
            .unwrap();
    }
}

fn neural_backprop(arena: &mut Vec<NeuralNode>, mut idx: usize, mut v: f32) {
    loop {
        arena[idx].visits    += 1;
        arena[idx].value_sum += v;
        v = -v;
        match arena[idx].parent { Some(p) => idx = p, None => break }
    }
}

pub fn neural_best_move(pos: &Chess, session: &mut Session) -> Option<Move> {
    let deadline = Instant::now() + Duration::from_millis(NEURAL_TIME);
    let mut arena: Vec<NeuralNode> = vec![NeuralNode::new(pos.clone(), None, None, 1.0)];

    let root_value = neural_expand(&mut arena, 0, session);
    neural_backprop(&mut arena, 0, root_value);

    let mut sims = 0u32;
    while Instant::now() < deadline && sims < NEURAL_SIMS {
        let leaf = neural_select(&arena);
        let value = if arena[leaf].pos.is_game_over() {
            match arena[leaf].pos.outcome() {
                Some(Outcome::Decisive { winner }) => {
                    if winner == flip_color(arena[leaf].pos.turn()) { 1.0 } else { -1.0 }
                }
                _ => 0.0,
            }
        } else {
            -neural_expand(&mut arena, leaf, session)
        };
        neural_backprop(&mut arena, leaf, value);
        sims += 1;
    }

    eprintln!("[neural] {} sims", sims);

    arena[0].children.iter()
        .max_by_key(|&&c| arena[c].visits)
        .and_then(|&c| arena[c].mv.clone())
}
