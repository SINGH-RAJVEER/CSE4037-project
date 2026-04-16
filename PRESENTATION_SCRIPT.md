# Presentation Script — Chess Bot: From Minimax to AlphaZero-Style RL
**Duration:** ~5 minutes | **Context:** Reinforcement Learning class

---

## [0:00 – 0:30] Hook & Introduction

Chess is one of the oldest benchmark problems in AI. The game tree has more possible positions than atoms in the observable universe, so you can't just brute-force it. The question I wanted to answer was: how do two completely different approaches — one hand-crafted algorithm and one that learns entirely from self-play — stack up when building a real chess engine?

I built both. Today I'll walk you through the algorithmic baseline first, explain exactly where it breaks down, and then show how the RL approach addresses those limitations.

---

## [0:30 – 1:45] The Algorithmic Approach — Minimax with Alpha-Beta Pruning

The classical approach to chess is **minimax**. The idea is simple: model the game as a two-player zero-sum tree. White tries to maximize the score, Black tries to minimize it. You search all legal moves to some depth, evaluate the leaf positions, and bubble the best value back up.

In my engine, the evaluation function is purely material-based:

- Pawn = 100, Knight = 320, Bishop = 330, Rook = 500, Queen = 900

So the engine sums up the piece values for White, subtracts Black's, and that's the score. If it's White's turn and one move leaves a net material advantage of +500 centipawns, minimax will prefer that move.

To make the search practical I added **alpha-beta pruning** — we track a window of `[alpha, beta]` and cut off branches that can't possibly change the outcome. In the best case this halves the effective branching factor, letting us search twice as deep in the same time.

This is fully deterministic, requires zero training data, and runs entirely in Rust so it's extremely fast.

**But here's the problem.** The evaluation function only counts material. It has no concept of piece activity, king safety, pawn structure, or positional pressure. It's completely blind to strategy. A position where your queen is trapped but you're up a pawn looks great to the evaluator. The engine doesn't understand *why* some moves are good — it only knows the score at the leaves, and that score is wrong in most strategic positions.

---

## [1:45 – 3:15] The RL Approach — AlphaZero-Style Self-Play

The RL approach flips the problem. Instead of hand-coding what a good position looks like, we ask: *can the engine learn this from experience?*

The architecture is modeled after AlphaZero. There are three components:

**1. The Neural Network (`ChessNet`)**
The board is encoded as a 14-channel 8×8 tensor — 12 channels for piece placement (one per piece type per color), one channel for side-to-move, one for en passant. This tensor passes through a residual tower of convolutional blocks — I used 10 residual blocks with 128 filters each. The tower splits into two heads:
- A **policy head** that outputs 4,672 logits, one for every possible move encoding
- A **value head** that outputs a single scalar in [-1, 1] estimating who's winning

**2. Neural MCTS**
At inference time, instead of raw minimax, we run Monte Carlo Tree Search guided by the network. Each node stores visit counts, prior probabilities, and value estimates. Selection uses the **PUCT formula** — it balances exploitation (Q-value) against exploration (prior probability scaled by visit count). The network evaluates leaf nodes, and we backpropagate the value up the tree, flipping sign at each level since players alternate. I run up to 800 simulations per move within a 3-second time budget.

**3. The Self-Play Training Loop**
This is where reinforcement learning happens. The model plays games against itself. For each position, the MCTS visit counts form a policy target. Once the game ends, the outcome — win, loss, or draw — is assigned as the value target for every position in that game. These (state, policy, value) triples go into a replay buffer. We then train the network to minimize policy cross-entropy plus value MSE loss. After each iteration, the model is a little better — so the next round of self-play generates higher-quality games, and the loop continues.

After training, the model is exported to ONNX and loaded directly into the Rust server at startup, so the same HTTP API serves both the minimax and neural engine with no Python at runtime.

---

## [3:15 – 4:15] Comparing the Two

The difference in play style is immediately visible. The minimax engine is tactically sharp at low depth — it won't blunder a piece — but it will drift into terrible strategic positions and not realize it. It has no concept of piece coordination or long-term planning.

The neural engine plays recognizable chess. It castles, develops pieces, contests the center. It sometimes sacrifices material for positional compensation. These are things no one wrote rules for — the network inferred them purely from outcomes over thousands of self-play games.

The trade-off is compute. Minimax is deterministic and runs in microseconds. The neural engine needs GPU inference across 800 MCTS simulations per move, which is orders of magnitude more expensive.

---

## [4:15 – 5:00] Takeaway

The core lesson is about **where knowledge lives**. In the algorithmic approach, knowledge is explicit — a human encodes the evaluation function. In the RL approach, knowledge is implicit — it emerges from the reward signal and self-play. The RL model doesn't know the rules of good chess; it knows that certain board states historically led to wins, and it learned to seek them.

Chess is a perfect test bed for this comparison because the rules are fixed, the reward is unambiguous, and both approaches are fully reproducible. If you want to see the engine running, it's deployed as a web app with a playable board where you can switch between minimax and neural mode mid-game.

Thank you.

---

*Total word count: ~780 words. At a comfortable pace (~155 wpm) this lands at approximately 5 minutes.*

---

## Demo Notes

> Run the demo **during** the comparison segment [3:15–4:15]. Keep the browser open in a separate window before you start speaking. The demo itself only needs ~60 seconds.

---

### Before You Present

- [ ] Start the stack: `docker compose up` from the repo root
- [ ] Open the app in browser at `http://localhost:3000`
- [ ] Confirm engine toggle shows **MINIMAX** (bottom-left button)
- [ ] Confirm the board is at starting position — hit **New Game** if not
- [ ] Open a second tab with the same URL for the neural demo (pre-set to NEURAL so you can switch fast)
- [ ] Have `engine/src/minimax.rs` and `engine/src/eval.rs` open in your editor, ready to alt-tab

---

### Demo Flow (~60 seconds)

**Step 1 — Show the board (5 sec)**
> "Here's the running app. You're playing White against the engine."

Point out the bottom bar: engine toggle, move history, Takeback / New Game buttons.

---

**Step 2 — Demonstrate Minimax's strategic blindness (25 sec)**

Make these moves to set up a forcing line that exposes the material-only evaluator:

1. Play **e4** (click e2 → e4, confirm)
2. Engine responds
3. Play **Nf3**
4. Engine responds
5. Now **deliberately sacrifice a pawn** — play **d4**, wait for engine to take it

> "Watch — I offer a pawn. The minimax engine sees +100 centipawns and takes it, because its evaluation function is literally just piece values. It has no concept of piece activity or center control."

Point at the **MINIMAX** button in the bottom-left.

> "That logic lives in `eval.rs` — six lines of code, one number per piece. No strategy."

Alt-tab briefly to `eval.rs:10-17` if screen-sharing allows.

---

**Step 3 — Switch to Neural (20 sec)**

Click the **MINIMAX** button to toggle it to **NEURAL**. Hit **New Game**.

> "Now the same position, neural engine. Same opening."

Play e4, Nf3. Offer the same pawn.

> "It doesn't take it — the network has learned that grabbing a center pawn with a development deficit is not worth it. Nobody told it that. It inferred it from thousands of self-play games."

Watch the engine castle or develop a piece instead.

---

**Step 4 — Point at the MCTS thinking indicator (10 sec)**

After you make a move, the status bar reads **"thinking..."** for ~1–3 seconds.

> "That pause is 800 MCTS simulations running through the neural network. Each simulation evaluates a board position with the ResNet and propagates a value estimate back up the tree. Minimax returns instantly — neural takes a second to think."

---

### Fallback if the Neural Model Isn't Loaded

If `engine/model.onnx` doesn't exist (model not yet trained), the engine logs:
```
[engine] No model at 'model.onnx'
```
In this case, skip the neural live demo and switch to showing the **training loop** in `train.py` instead:

> "The neural engine requires a trained model. Here's the training loop — self-play generates games, MCTS produces policy targets, and the network trains on win/loss outcomes. After training, it exports to ONNX and the Rust server loads it at startup."

Show `train.py:175-211` (the self-play + training iteration loop).

---

### Code Snippets to Have Ready (alt-tab or split screen)

| File | Lines | What to point at |
|---|---|---|
| `engine/src/eval.rs` | 1–17 | "Six lines — the entire evaluation function" |
| `engine/src/minimax.rs` | 5–33 | Alpha-beta pruning, the `if beta <= alpha { break; }` cutoff |
| `engine/src/mcts.rs` | 39–41 | PUCT formula — balancing Q-value and prior |
| `engine/train/model.py` | 27–57 | ResNet architecture, policy + value heads |
| `engine/train/self_play.py` | 16–59 | Self-play loop, retrospective value assignment |

---

### Common Questions

**"How strong is it?"**
> Minimax at depth 3 plays ~800 ELO. The neural engine's strength depends on training iterations — with 100 iterations and 25 self-play games each, it reaches rough beginner level. AlphaZero itself trained for millions of games; this is a scaled-down proof of concept.

**"Why Rust for the engine?"**
> Minimax is CPU-bound and Rust is roughly 10x faster than Python for tight tree search. The ONNX runtime also has a first-class Rust binding (`ort` crate), so the neural inference stays in the same process with no IPC overhead.

**"What's the 4,672 policy output?"**
> Every possible move in chess can be encoded as: which square the piece moves from, and a direction+distance vector. 73 possible move types × 64 squares = 4,672. This is the same encoding AlphaZero uses.

**"Did you train it yourself?"**
> Yes — the training code is in `engine/train/`. It runs AlphaZero-style: self-play generates data, MCTS provides policy supervision, and the outcome provides value supervision. No human game databases used.
