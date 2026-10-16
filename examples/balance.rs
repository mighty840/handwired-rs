//! Balancing a small game through its own rules.
//!
//! Three units fight dice skirmishes. The rules are one ordinary function,
//! compiled into an exact circuit. Each tuning step counts every matchup's win
//! rate exactly, by running the circuit on a fixed set of training fights, and
//! takes the direction to move from the circuit's backward pass: the gradient
//! of each fight's margin with respect to both units' damage and health. A few
//! Gauss-Newton steps move the six numbers until every matchup is close to
//! 50/50, checked on fights the tuning never saw, each against the plain
//! function.
//!
//! cargo run --release --example balance

use handwired_rs::{BackpropOptions, BooleanGradient, IndexGradient, handwired};

const VOLLEYS: usize = 10;

/// One skirmish: ten volleys in which both units shoot. A shot hits on a roll
/// of at least `hits_on` and deals the shooter's damage minus the target's
/// armor, doubled on a 6. The skirmish always lasts all ten volleys, and health
/// may go below zero. Returns the margin: positive when unit A keeps the larger
/// share of its health, negative when B does.
#[handwired]
#[allow(clippy::too_many_arguments)]
fn skirmish(
    #[range(2, 20)] damage_a: f32,
    #[range(2, 20)] damage_b: f32,
    #[range(1, 200)] health_a: f32,
    #[range(1, 200)] health_b: f32,
    #[range(0, 2)] armor_a: f32,
    #[range(0, 2)] armor_b: f32,
    #[range(2, 6)] hits_on_a: u8,
    #[range(2, 6)] hits_on_b: u8,
    #[range(1, 6)] rolls_a: [u8; 10],
    #[range(1, 6)] rolls_b: [u8; 10],
) -> f32 {
    let mut taken_a: f32 = 0.0;
    let mut taken_b: f32 = 0.0;
    for volley in 0..10 {
        if rolls_a[volley] >= hits_on_a {
            let mut hit = damage_a - armor_b;
            if rolls_a[volley] == 6 {
                hit *= 2.0;
            }
            taken_b += hit;
        }
        if rolls_b[volley] >= hits_on_b {
            let mut hit = damage_b - armor_a;
            if rolls_b[volley] == 6 {
                hit *= 2.0;
            }
            taken_a += hit;
        }
    }
    let left_a = health_a - taken_a;
    let left_b = health_b - taken_b;
    // left_a / health_a - left_b / health_b, times health_a × health_b.
    left_a * health_b - left_b * health_a
}

#[derive(Copy, Clone)]
struct Unit {
    name: &'static str,
    damage: f32,
    health: f32,
    armor: f32,
    hits_on: u8,
}

const UNITS: [Unit; 3] = [
    Unit { name: "Knight", damage: 6.0, health: 60.0, armor: 2.0, hits_on: 4 },
    Unit { name: "Archer", damage: 4.0, health: 45.0, armor: 0.0, hits_on: 3 },
    Unit { name: "Mage", damage: 9.0, health: 36.0, armor: 1.0, hits_on: 5 },
];
const MATCHUPS: [(usize, usize); 3] = [(0, 1), (1, 2), (2, 0)];

type Fight = (f32, f32, f32, f32, f32, f32, u8, u8, [u8; VOLLEYS], [u8; VOLLEYS]);

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn dice(&mut self) -> [u8; VOLLEYS] {
        std::array::from_fn(|_| 1 + (self.next() % 6) as u8)
    }
}

/// Dice for `count` fights per matchup.
fn roll(rng: &mut Rng, count: usize) -> Vec<Vec<([u8; VOLLEYS], [u8; VOLLEYS])>> {
    MATCHUPS.iter().map(|_| (0..count).map(|_| (rng.dice(), rng.dice())).collect()).collect()
}

/// What gradient descent tunes: each unit's damage and health.
type Stats = [[f32; 2]; 3];
const DAMAGE: usize = 0;
const HEALTH: usize = 1;
/// The bounds the tuning keeps each number within, inside the declared ranges.
const BOUNDS: [(f64, f64); 2] = [(2.0, 20.0), (10.0, 200.0)];

fn fight(stats: &Stats, (a, b): (usize, usize), rolls: ([u8; VOLLEYS], [u8; VOLLEYS])) -> Fight {
    let (unit_a, unit_b) = (UNITS[a], UNITS[b]);
    (
        stats[a][DAMAGE],
        stats[b][DAMAGE],
        stats[a][HEALTH],
        stats[b][HEALTH],
        unit_a.armor,
        unit_b.armor,
        unit_a.hits_on,
        unit_b.hits_on,
        rolls.0,
        rolls.1,
    )
}

/// Exact win rates of the first unit in each matchup, a draw counting half, from the circuit on every core; each
/// result is also checked against the plain function.
fn win_rates(
    compiled: &handwired_rs::dsl::Compiled<Fight, f32>,
    stats: &Stats,
    dice: &[Vec<([u8; VOLLEYS], [u8; VOLLEYS])>],
) -> [f64; 3] {
    std::array::from_fn(|matchup| {
        let fights: Vec<Fight> = dice[matchup].iter().map(|&rolls| fight(stats, MATCHUPS[matchup], rolls)).collect();
        let margins = compiled.run_batch(&fights).unwrap();
        let mut score = 0.0;
        for (fight, margin) in fights.iter().zip(margins) {
            let (a, b, c, d, e, f, g, h, i, j) = *fight;
            assert_eq!(margin, skirmish(a, b, c, d, e, f, g, h, i, j), "the circuit disagrees with skirmish()");
            score += if margin > 0.0 {
                1.0
            } else if margin == 0.0 {
                0.5
            } else {
                0.0
            };
        }
        score / fights.len() as f64
    })
}

/// Solves a 3 × 3 linear system by Gaussian elimination with partial pivoting.
fn solve3(mut matrix: [[f64; 3]; 3], mut right: [f64; 3]) -> [f64; 3] {
    for column in 0..3 {
        let pivot = (column..3).max_by(|&x, &y| matrix[x][column].abs().total_cmp(&matrix[y][column].abs())).unwrap();
        matrix.swap(column, pivot);
        right.swap(column, pivot);
        for row in column + 1..3 {
            let factor = matrix[row][column] / matrix[column][column];
            let pivot_row = matrix[column];
            for (entry, pivot_entry) in matrix[row].iter_mut().zip(pivot_row).skip(column) {
                *entry -= factor * pivot_entry;
            }
            right[row] -= factor * right[column];
        }
    }
    let mut solution = [0.0; 3];
    for row in (0..3).rev() {
        let known: f64 = (row + 1..3).map(|k| matrix[row][k] * solution[k]).sum();
        solution[row] = (right[row] - known) / matrix[row][row];
    }
    solution
}

fn print_rates(label: &str, stats: &Stats, rates: &[f64; 3]) {
    let units: Vec<String> = UNITS
        .iter()
        .zip(stats)
        .map(|(unit, stats)| format!("{} {:.2} dmg / {:.1} hp", unit.name, stats[DAMAGE], stats[HEALTH]))
        .collect();
    let matchups: Vec<String> = MATCHUPS
        .iter()
        .zip(rates)
        .map(|(&(a, b), rate)| format!("{} beats {} {:>5.1}%", UNITS[a].name, UNITS[b].name, 100.0 * rate))
        .collect();
    println!(
        "{label:<8} {}
         {}",
        units.join(", "),
        matchups.join(" | ")
    );
}

fn main() {
    let compiled = skirmish_circuit();
    let report = compiled.verify(20_000).unwrap();
    println!(
        "skirmish: {} live operations in {} layers, verified on {} sampled fights\n",
        compiled.program.live_operation_count(),
        compiled.program.layers.len(),
        report.checked
    );
    // The rolls only reach comparisons, and the gradient for damage and health never passes through one; these
    // rules only let the backward pass run through the comparisons on its way to the dice.
    let options =
        BackpropOptions { boolean_gradient: BooleanGradient::StraightThrough, index_gradient: IndexGradient::Zero };

    let mut rng = Rng(0xBA1A_4CE5_D1CE_5EED);
    let held_out = roll(&mut rng, 20_000);
    let start: Stats = UNITS.map(|unit| [unit.damage, unit.health]);
    let mut stats = start;
    let before = win_rates(&compiled, &stats, &held_out);

    // Gauss-Newton on the three win rates, over the six stats. Each step counts the exact win rates on a fixed set
    // of training fights (the residuals), and takes their slopes from the circuit's backward pass: the slope of a
    // smoothed win rate, the mean of sigmoid(margin / T), with T a tenth of the typical margin. Then it moves the
    // stats by the smallest change that would bring all three rates to 50%.
    let training = roll(&mut rng, 16_384);
    // Fights with the same hits change winner at the same stats, so the rates move in steps of a few points and 2%
    // is about as close as they get. The trust radius halves whenever a step doesn't help.
    let (mut radius, mut best, mut best_stats) = (1.0f64, f64::INFINITY, stats);
    for step in 0..40 {
        let rates = win_rates(&compiled, &stats, &training);
        let residuals = rates.map(|rate| rate - 0.5);
        println!(
            "step {step:>2}: win rates {:.1}% / {:.1}% / {:.1}%",
            100.0 * rates[0],
            100.0 * rates[1],
            100.0 * rates[2]
        );
        let worst = residuals.iter().fold(0.0f64, |worst, residual| worst.max(residual.abs()));
        if worst < best {
            (best, best_stats) = (worst, stats);
        } else {
            radius /= 2.0;
            stats = best_stats;
        }
        if best < 0.02 || radius < 0.05 {
            break;
        }
        // jacobian[matchup][unit * 2 + stat]: how the matchup's win rate moves with each stat.
        let mut jacobian = [[0.0f64; 6]; 3];
        for (matchup, &(a, b)) in MATCHUPS.iter().enumerate() {
            let fights: Vec<Fight> = training[matchup].iter().map(|&rolls| fight(&stats, (a, b), rolls)).collect();
            let margins = compiled.run_batch(&fights).unwrap();
            let temperature =
                0.1 * margins.iter().map(|margin| margin.abs() as f64).sum::<f64>() / margins.len() as f64;
            // Each fight's d(smoothed rate)/d(margin) weights the gradient of its margin.
            let weights: Vec<f32> = margins
                .iter()
                .map(|&margin| {
                    let sigmoid = 1.0 / (1.0 + (-margin as f64 / temperature).exp());
                    (sigmoid * (1.0 - sigmoid) / temperature / fights.len() as f64) as f32
                })
                .collect();
            let gradients = compiled.gradients(&fights, 0, Some(&weights), options).unwrap();
            // Source cells 0..4 are damage_a, damage_b, health_a and health_b.
            jacobian[matchup][a * 2 + DAMAGE] += gradients[0];
            jacobian[matchup][b * 2 + DAMAGE] += gradients[1];
            jacobian[matchup][a * 2 + HEALTH] += gradients[2];
            jacobian[matchup][b * 2 + HEALTH] += gradients[3];
        }
        // Health is measured in tens, so a point of damage and ten of health count alike in "smallest change".
        let scale = [1.0, 10.0];
        let scaled: [[f64; 6]; 3] = jacobian.map(|row| std::array::from_fn(|k| row[k] * scale[k % 2]));
        // Δ = -Jᵀ (J Jᵀ + λI)⁻¹ r, the damped least-norm step.
        let mut normal = [[0.0f64; 3]; 3];
        for row in 0..3 {
            for column in 0..3 {
                normal[row][column] = (0..6).map(|k| scaled[row][k] * scaled[column][k]).sum::<f64>();
            }
            normal[row][row] += 1e-4;
        }
        let multipliers = solve3(normal, residuals);
        let step: [f64; 6] = std::array::from_fn(|k| -(0..3).map(|row| scaled[row][k] * multipliers[row]).sum::<f64>());
        // A trust region: at most `radius` points of damage or ten times that of health per step, since the win
        // rates are only locally linear.
        let largest = step.iter().fold(0.0f64, |largest, change| largest.max(change.abs()));
        let shrink = (largest / radius).max(1.0);
        for unit in 0..3 {
            for stat in [DAMAGE, HEALTH] {
                let change = step[unit * 2 + stat] / shrink * scale[stat];
                let (low, high) = BOUNDS[stat];
                stats[unit][stat] = (stats[unit][stat] as f64 + change).clamp(low, high) as f32;
            }
        }
    }

    stats = best_stats;
    println!();
    print_rates("before", &start, &before);
    let after = win_rates(&compiled, &stats, &held_out);
    print_rates("tuned", &stats, &after);
    println!(
        "\ntuned on 16,384 training fights per matchup; win rates above are counted on 20,000 other fights per matchup, \
         and every fight matches skirmish()"
    );
}
