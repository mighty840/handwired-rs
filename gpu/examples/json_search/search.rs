//! Regex search over JSON documents as a loop program, after handwired's JSON search (src/json-search): a pattern
//! compiled into an automaton whose tables are data, three steps written as ordinary Rust functions, and a reference
//! written from the definition of what the search finds.
//!
//! What the search finds, as in handwired: every non-overlapping, non-empty match, leftmost, then longest. From a scan
//! position, the longest match that starts there is counted and the scan continues at its end; where no match starts,
//! the scan moves one character on. A program can't go back, so it reads a document twice: backward, to know at each
//! position from which states an accepting state is still reachable, then forward, extending the match in progress
//! while it can still end in an accepting state, and starting one where it can't.
//!
//! Differences from handwired: documents are ASCII, at most 256 bytes, where handwired reads 512 code points; the
//! program holds patterns of up to 8 states and 16 character classes, where handwired has three sizes up to 32
//! states; and the string scope tracks quotes and escapes, so it reads keys and string values alike, where
//! handwired's JSON highlighter also tells keys from values.

#![allow(dead_code)]

use handwired_rs::handwired;
use handwired_rs_gpu::looped::{Binding, Direction, FieldKind, LoopProgram, Sink, Step};

/// Characters per document: a document is this many cells, its bytes and then -1.
pub const POSITIONS: usize = 256;
/// States and character classes the program holds.
pub const STATES: usize = 8;
pub const CLASSES: usize = 16;
/// Tables have a column past the last class, for bytes that no state reads and for -1, which read nothing.
pub const COLUMNS: usize = CLASSES + 1;

// ---- The pattern ------------------------------------------------------------------------------------------------

/// The ASCII bytes a step into a state reads.
type Set = [bool; 128];

/// A pattern as data: the class of each byte, and for each pair of states and each class whether a step from the
/// first state reads a byte of that class and goes to the second. State 0 is where every match starts.
#[derive(Clone, Debug)]
pub struct Pattern {
    pub source: String,
    pub states: usize,
    pub classes: usize,
    /// `class_of[byte + 1]`, with `class_of[0]` for -1; `CLASSES` for a byte no state reads.
    pub class_of: [u8; 129],
    /// `transitions[from][to][class]`.
    pub transitions: [[[bool; COLUMNS]; STATES]; STATES],
    pub accepting: [bool; STATES],
    /// The automaton, for the reference: each state's targets and the bytes a step into it reads.
    targets: Vec<Vec<usize>>,
    sets: Vec<Set>,
}

#[derive(Debug, PartialEq)]
pub struct PatternError {
    pub message: String,
    pub position: usize,
}

enum Node {
    Atom(Set),
    Sequence(Vec<Node>),
    Alternation(Vec<Node>),
    Repeat { body: Box<Node>, optional: bool, repeats: bool },
}

/// The pattern language of handwired's JSON search, on ASCII: literal characters, `.` (any but `\n` and `\r`), `\d`,
/// `\w`, a backslash before a syntax character, classes with ranges and negation (`[a-z_]`, `[^0-9]`), `?`, `*`, `+`,
/// `|` and groups.
pub fn compile(source: &str) -> Result<Pattern, PatternError> {
    let bytes = source.as_bytes();
    if let Some(position) = bytes.iter().position(|&byte| byte >= 128) {
        return Err(PatternError { message: "the pattern is ASCII here".into(), position });
    }
    let mut parser = Parser { bytes, position: 0 };
    let node = parser.alternation()?;
    if parser.position < bytes.len() {
        return Err(parser.fail("this ) closes no group"));
    }
    // Glushkov's construction: state 0 is the start, state k + 1 the pattern's k-th atom, so every step into a state
    // reads that atom's bytes and the automaton has no empty steps.
    let mut sets = Vec::new();
    let mut follow: Vec<Vec<usize>> = Vec::new();
    let summary = summarize(&node, &mut sets, &mut follow);
    let states = sets.len() + 1;
    let mut targets = vec![sorted(&summary.first)];
    targets.extend(follow.iter().map(|atoms| sorted(atoms)));
    let mut accepting = vec![summary.nullable];
    accepting.extend((0..sets.len()).map(|atom| summary.last.contains(&atom)));
    let mut state_sets = vec![[false; 128]];
    state_sets.extend(sets.iter().copied());
    if states > STATES {
        return Err(PatternError {
            message: format!("the pattern needs {states} states, and the program holds {STATES}"),
            position: 0,
        });
    }
    // Bytes that every step reads the same way share a class.
    let mut signatures: Vec<Vec<bool>> = Vec::new();
    let mut class_of = [CLASSES as u8; 129];
    for byte in 0..128 {
        let reads: Vec<bool> = state_sets.iter().map(|set| set[byte]).collect();
        if !reads.iter().any(|&read| read) {
            continue;
        }
        let class = signatures.iter().position(|signature| *signature == reads).unwrap_or_else(|| {
            signatures.push(reads.clone());
            signatures.len() - 1
        });
        if class >= CLASSES {
            return Err(PatternError {
                message: format!("the pattern needs more than {CLASSES} character classes"),
                position: 0,
            });
        }
        class_of[byte + 1] = class as u8;
    }
    let mut transitions = [[[false; COLUMNS]; STATES]; STATES];
    for (from, list) in targets.iter().enumerate() {
        for &to in list {
            for (class, signature) in signatures.iter().enumerate() {
                transitions[from][to][class] = signature[to];
            }
        }
    }
    let mut accepting_cells = [false; STATES];
    accepting_cells[..states].copy_from_slice(&accepting);
    Ok(Pattern {
        source: source.to_string(),
        states,
        classes: signatures.len(),
        class_of,
        transitions,
        accepting: accepting_cells,
        targets,
        sets: state_sets,
    })
}

struct Summary {
    nullable: bool,
    first: Vec<usize>,
    last: Vec<usize>,
}

fn sorted(atoms: &[usize]) -> Vec<usize> {
    let mut states: Vec<usize> = atoms.iter().map(|atom| atom + 1).collect();
    states.sort_unstable();
    states.dedup();
    states
}

fn union(left: &[usize], right: &[usize]) -> Vec<usize> {
    let mut all = left.to_vec();
    all.extend(right.iter().filter(|atom| !left.contains(atom)));
    all
}

fn summarize(node: &Node, sets: &mut Vec<Set>, follow: &mut Vec<Vec<usize>>) -> Summary {
    match node {
        Node::Atom(set) => {
            let atom = sets.len();
            sets.push(*set);
            follow.push(Vec::new());
            Summary { nullable: false, first: vec![atom], last: vec![atom] }
        }
        Node::Sequence(parts) => {
            let mut summary = Summary { nullable: true, first: Vec::new(), last: Vec::new() };
            for part in parts {
                let next = summarize(part, sets, follow);
                for &atom in &summary.last {
                    follow[atom] = union(&follow[atom], &next.first);
                }
                summary = Summary {
                    nullable: summary.nullable && next.nullable,
                    first: if summary.nullable { union(&summary.first, &next.first) } else { summary.first },
                    last: if next.nullable { union(&summary.last, &next.last) } else { next.last },
                };
            }
            summary
        }
        Node::Alternation(options) => {
            let mut summary = Summary { nullable: false, first: Vec::new(), last: Vec::new() };
            for option in options {
                let next = summarize(option, sets, follow);
                summary.nullable |= next.nullable;
                summary.first = union(&summary.first, &next.first);
                summary.last = union(&summary.last, &next.last);
            }
            summary
        }
        Node::Repeat { body, optional, repeats } => {
            let body = summarize(body, sets, follow);
            if *repeats {
                for &atom in &body.last {
                    follow[atom] = union(&follow[atom], &body.first);
                }
            }
            Summary { nullable: body.nullable || *optional, first: body.first, last: body.last }
        }
    }
}

struct Parser<'a> {
    bytes: &'a [u8],
    position: usize,
}

const SYNTAX: &[u8] = b"^$\\.*+?()[]{}|/";

impl Parser<'_> {
    fn fail(&self, message: &str) -> PatternError {
        PatternError { message: message.to_string(), position: self.position }
    }

    fn is(&self, byte: u8) -> bool {
        self.bytes.get(self.position) == Some(&byte)
    }

    fn alternation(&mut self) -> Result<Node, PatternError> {
        let mut options = vec![self.sequence()?];
        while self.is(b'|') {
            self.position += 1;
            options.push(self.sequence()?);
        }
        Ok(if options.len() == 1 { options.pop().unwrap() } else { Node::Alternation(options) })
    }

    fn sequence(&mut self) -> Result<Node, PatternError> {
        let mut parts = Vec::new();
        while self.position < self.bytes.len() && !self.is(b'|') && !self.is(b')') {
            parts.push(self.repeat()?);
        }
        Ok(if parts.len() == 1 { parts.pop().unwrap() } else { Node::Sequence(parts) })
    }

    fn repeat(&mut self) -> Result<Node, PatternError> {
        let body = self.atom()?;
        let quantifier = match self.bytes.get(self.position) {
            Some(&byte @ (b'?' | b'*' | b'+')) => byte,
            _ => return Ok(body),
        };
        self.position += 1;
        if self.is(b'?') {
            return Err(self.fail("lazy quantifiers aren't in this language"));
        }
        if self.is(b'*') || self.is(b'+') {
            return Err(self.fail("nothing to repeat"));
        }
        Ok(Node::Repeat { body: Box::new(body), optional: quantifier != b'+', repeats: quantifier != b'?' })
    }

    fn atom(&mut self) -> Result<Node, PatternError> {
        let at = self.position;
        let byte = self.bytes[self.position];
        self.position += 1;
        Ok(Node::Atom(match byte {
            b'(' => {
                if self.is(b'?') {
                    return Err(PatternError { message: "groups such as (?: aren't supported".into(), position: at });
                }
                let inner = self.alternation()?;
                if !self.is(b')') {
                    return Err(PatternError { message: "this group is never closed".into(), position: at });
                }
                self.position += 1;
                return Ok(inner);
            }
            b'[' => self.class(at)?,
            b'.' => {
                let mut set = [true; 128];
                set[b'\n' as usize] = false;
                set[b'\r' as usize] = false;
                set
            }
            b'\\' => self.escape(at, false)?.0,
            b'?' | b'*' | b'+' => return Err(PatternError { message: "nothing to repeat".into(), position: at }),
            b'^' | b'$' => {
                return Err(PatternError { message: "the anchors ^ and $ aren't supported".into(), position: at });
            }
            b'{' | b'}' | b']' => {
                return Err(PatternError { message: "this character needs a backslash".into(), position: at });
            }
            byte => single(byte),
        }))
    }

    /// After a backslash: the set, and the byte where it's one, for a range's ends.
    fn escape(&mut self, at: usize, in_class: bool) -> Result<(Set, Option<u8>), PatternError> {
        let Some(&escaped) = self.bytes.get(self.position) else {
            return Err(PatternError { message: "a pattern can't end with a backslash".into(), position: at });
        };
        self.position += 1;
        match escaped {
            b'd' => Ok((set_of(|byte| byte.is_ascii_digit()), None)),
            b'w' => Ok((set_of(|byte| byte.is_ascii_alphanumeric() || byte == b'_'), None)),
            byte if SYNTAX.contains(&byte) || in_class && byte == b'-' => Ok((single(byte), Some(byte))),
            _ => Err(PatternError { message: "this escape isn't supported".into(), position: at }),
        }
    }

    fn class(&mut self, at: usize) -> Result<Set, PatternError> {
        let negated = self.is(b'^');
        if negated {
            self.position += 1;
        }
        let mut set = [false; 128];
        while !self.is(b']') {
            if self.position >= self.bytes.len() {
                return Err(PatternError { message: "this set of characters is never closed".into(), position: at });
            }
            let low = self.class_atom()?;
            let next = self.bytes.get(self.position + 1).copied();
            if self.is(b'-') && next.is_some() && next != Some(b']') {
                let dash = self.position;
                self.position += 1;
                let high = self.class_atom()?;
                let (Some(low), Some(high)) = (low.1, high.1) else {
                    return Err(PatternError {
                        message: "a range needs one character at each end".into(),
                        position: dash,
                    });
                };
                if low > high {
                    return Err(PatternError { message: "this range is out of order".into(), position: dash });
                }
                for byte in low..=high {
                    set[byte as usize] = true;
                }
            } else {
                for (read, &also) in set.iter_mut().zip(&low.0) {
                    *read |= also;
                }
            }
        }
        self.position += 1;
        if negated {
            set = set.map(|read| !read);
        }
        Ok(set)
    }

    fn class_atom(&mut self) -> Result<(Set, Option<u8>), PatternError> {
        let at = self.position;
        let byte = self.bytes[self.position];
        self.position += 1;
        if byte == b'\\' { self.escape(at, true) } else { Ok((single(byte), Some(byte))) }
    }
}

fn single(byte: u8) -> Set {
    let mut set = [false; 128];
    set[byte as usize] = true;
    set
}

fn set_of(test: impl Fn(u8) -> bool) -> Set {
    std::array::from_fn(|byte| test(byte as u8))
}

// ---- The steps ------------------------------------------------------------------------------------------------------

/// Forward over a document: whether each character is inside a string, its quotes included, and not part of an
/// escape, so that a search of strings reads no match across an escape such as `\n`.
#[handwired]
pub fn tokenize_step(#[range(-1, 127)] byte: i16, in_string: bool, escaped: bool) -> (bool, bool, bool) {
    let quote = byte == 34;
    let backslash = byte == 92;
    if escaped {
        (false, true, false)
    } else if in_string {
        (!backslash, !quote, backslash)
    } else {
        (quote, quote, false)
    }
}

/// Backward over a document: the character's class, and `reach[s]`, whether some path from state `s` reads this
/// character, maybe more, and ends in an accepting state. `reach_after` is the next character's. A character that
/// the search doesn't read, outside the scope or past the text, is -1, whose class no step reads.
#[handwired]
#[allow(clippy::needless_range_loop)]
pub fn reverse_step(
    #[range(-1, 127)] byte: i16,
    searchable: bool,
    everything: bool,
    #[range(0, 16)] class_of: [u8; 129],
    transitions: [[[bool; 17]; 8]; 8],
    accepting: [bool; 8],
    reach_after: [bool; 8],
) -> (u8, [bool; 8], [bool; 8]) {
    let read = if everything || searchable { byte } else { -1 };
    let class = class_of[(read + 1) as usize];
    let mut reach = [false; 8];
    for from in 0..8 {
        let mut any = false;
        for to in 0..8 {
            any = any || (transitions[from][to][class as usize] && (accepting[to] || reach_after[to]));
        }
        reach[from] = any;
    }
    (class, reach, reach)
}

/// Forward over a document: a match in progress continues through this character while some state it reached can
/// still end in an accepting state here or later; otherwise a match starts here if one can. `count` is the matches
/// started so far.
#[handwired]
#[allow(clippy::needless_range_loop)]
pub fn forward_step(
    #[range(0, 16)] class: u8,
    reach: [bool; 8],
    transitions: [[[bool; 17]; 8]; 8],
    reached: [bool; 8],
    #[range(0, 256)] count: u16,
) -> ([bool; 8], u16) {
    let mut continuing = [false; 8];
    let mut any = false;
    for state in 0..8 {
        continuing[state] = reached[state] && reach[state];
        any = any || continuing[state];
    }
    let start = !any && reach[0];
    let mut live = continuing;
    live[0] = continuing[0] || start;
    let mut next = [false; 8];
    for to in 0..8 {
        let mut entered = false;
        for from in 0..8 {
            entered = entered || (live[from] && transitions[from][to][class as usize]);
        }
        next[to] = entered;
    }
    (next, if start { count + 1 } else { count })
}

// ---- The loop program -----------------------------------------------------------------------------------------------

/// Shared cells: each byte's class, the transitions, the accepting states, and whether the search reads everything.
pub const CLASS_OF: usize = 0;
pub const TRANSITIONS: usize = CLASS_OF + 129;
pub const ACCEPTING: usize = TRANSITIONS + STATES * STATES * COLUMNS;
pub const EVERYTHING: usize = ACCEPTING + STATES;
pub const SHARED: usize = EVERYTHING + 1;

/// Registers: the tokenizer's two bits, `reach_after`, `reached` and the count.
const IN_STRING: usize = 0;
const ESCAPED: usize = 1;
const REACH_AFTER: usize = 2;
const REACHED: usize = REACH_AFTER + STATES;
const COUNT: usize = REACHED + STATES;
/// Fields: whether the search reads each character, its class, and `reach`.
const SEARCHABLE: usize = 0;
const CLASS: usize = 1;
const REACH: usize = 2;

/// The search as a loop program: the tokenizer forward, the reverse step backward, then the forward step. Its one
/// output is each document's match count.
pub fn loop_program() -> LoopProgram {
    let shared = |start: usize, count: usize| (start..start + count).map(Binding::Shared);
    let tokenize = Step {
        program: tokenize_step_circuit().program,
        direction: Direction::Forward,
        sources: vec![Binding::Position, Binding::Register(IN_STRING), Binding::Register(ESCAPED)],
        sinks: vec![Sink::Field(SEARCHABLE), Sink::Register(IN_STRING), Sink::Register(ESCAPED)],
    };
    let mut sources = vec![Binding::Position, Binding::Field(SEARCHABLE), Binding::Shared(EVERYTHING)];
    sources.extend(shared(CLASS_OF, 129));
    sources.extend(shared(TRANSITIONS, STATES * STATES * COLUMNS));
    sources.extend(shared(ACCEPTING, STATES));
    sources.extend((0..STATES).map(|state| Binding::Register(REACH_AFTER + state)));
    let mut sinks = vec![Sink::Field(CLASS)];
    sinks.extend((0..STATES).map(|state| Sink::Field(REACH + state)));
    sinks.extend((0..STATES).map(|state| Sink::Register(REACH_AFTER + state)));
    let reverse = Step { program: reverse_step_circuit().program, direction: Direction::Backward, sources, sinks };
    let mut sources = vec![Binding::Field(CLASS)];
    sources.extend((0..STATES).map(|state| Binding::Field(REACH + state)));
    sources.extend(shared(TRANSITIONS, STATES * STATES * COLUMNS));
    sources.extend((0..STATES).map(|state| Binding::Register(REACHED + state)));
    sources.push(Binding::Register(COUNT));
    let mut sinks: Vec<Sink> = (0..STATES).map(|state| Sink::Register(REACHED + state)).collect();
    sinks.push(Sink::Register(COUNT));
    let forward = Step { program: forward_step_circuit().program, direction: Direction::Forward, sources, sinks };
    let mut fields = vec![FieldKind::Bit, FieldKind::Float];
    fields.extend([FieldKind::Bit; STATES]);
    LoopProgram {
        positions: POSITIONS,
        registers: vec![0.0; COUNT + 1],
        fields,
        shared_size: SHARED,
        steps: vec![tokenize, reverse, forward],
        outputs: vec![COUNT],
    }
}

/// The shared cells for a pattern, searching everything or only inside strings.
pub fn shared_cells(pattern: &Pattern, everything: bool) -> Vec<f32> {
    let mut cells = vec![0f32; SHARED];
    for (index, &class) in pattern.class_of.iter().enumerate() {
        cells[CLASS_OF + index] = class as f32;
    }
    for from in 0..STATES {
        for to in 0..STATES {
            for class in 0..COLUMNS {
                cells[TRANSITIONS + (from * STATES + to) * COLUMNS + class] =
                    pattern.transitions[from][to][class] as u8 as f32;
            }
        }
    }
    for state in 0..STATES {
        cells[ACCEPTING + state] = pattern.accepting[state] as u8 as f32;
    }
    cells[EVERYTHING] = everything as u8 as f32;
    cells
}

/// A document's cells: its bytes, then -1.
pub fn document_cells(text: &[u8], cells: &mut Vec<f32>) {
    assert!(text.len() <= POSITIONS && text.is_ascii(), "a document is at most {POSITIONS} ASCII bytes");
    cells.extend(text.iter().map(|&byte| byte as f32));
    cells.extend(std::iter::repeat_n(-1.0, POSITIONS - text.len()));
}

/// The steps' own Rust functions, run in the same loop: what the circuits compute, by construction.
pub fn native_count(pattern: &Pattern, text: &[u8], everything: bool) -> u16 {
    let mut cells = Vec::new();
    document_cells(text, &mut cells);
    let bytes: Vec<i16> = cells.iter().map(|&cell| cell as i16).collect();
    let mut searchable = [false; POSITIONS];
    let (mut in_string, mut escaped) = (false, false);
    for position in 0..POSITIONS {
        (searchable[position], in_string, escaped) = tokenize_step(bytes[position], in_string, escaped);
    }
    let mut classes = [0u8; POSITIONS];
    let mut reaches = [[false; STATES]; POSITIONS];
    let mut reach_after = [false; STATES];
    for position in (0..POSITIONS).rev() {
        let (class, reach, after) = reverse_step(
            bytes[position],
            searchable[position],
            everything,
            pattern.class_of,
            pattern.transitions,
            pattern.accepting,
            reach_after,
        );
        (classes[position], reaches[position], reach_after) = (class, reach, after);
    }
    let (mut reached, mut count) = ([false; STATES], 0u16);
    for position in 0..POSITIONS {
        (reached, count) = forward_step(classes[position], reaches[position], pattern.transitions, reached, count);
    }
    count
}

// ---- The reference --------------------------------------------------------------------------------------------------

/// The matches, as (start, end), from the definition and the automaton alone, as handwired's `automatonMatches`: from
/// each scan position, run the automaton until no state is left, and keep the last position where an accepting state
/// was reached. With `everything` false, only characters inside strings, outside escapes, are read.
pub fn reference_matches(pattern: &Pattern, text: &[u8], everything: bool) -> Vec<(usize, usize)> {
    let readable = readable(text, everything);
    let reads = |state: usize, index: usize| readable[index] && pattern.sets[state][text[index] as usize];
    let mut matches = Vec::new();
    let mut position = 0;
    while position < text.len() {
        let mut states = vec![0];
        let mut longest = None;
        let mut index = position;
        while index < text.len() && !states.is_empty() {
            let mut next: Vec<usize> = Vec::new();
            for &state in &states {
                for &target in &pattern.targets[state] {
                    if reads(target, index) && !next.contains(&target) {
                        next.push(target);
                    }
                }
            }
            states = next;
            if states.iter().any(|&state| pattern.accepting[state]) {
                longest = Some(index + 1);
            }
            index += 1;
        }
        match longest {
            Some(end) => {
                matches.push((position, end));
                position = end;
            }
            None => position += 1,
        }
    }
    matches
}

/// Which characters a search reads: all, or those inside strings, quotes included, outside escapes. Written apart
/// from the tokenizer step.
fn readable(text: &[u8], everything: bool) -> Vec<bool> {
    if everything {
        return vec![true; text.len()];
    }
    let mut readable = vec![false; text.len()];
    let mut index = 0;
    while index < text.len() {
        if text[index] != b'"' {
            index += 1;
            continue;
        }
        // A string from this quote to the next unescaped one, or to the end of the text.
        readable[index] = true;
        index += 1;
        while index < text.len() {
            match text[index] {
                b'\\' => index += 2,
                b'"' => {
                    readable[index] = true;
                    index += 1;
                    break;
                }
                _ => {
                    readable[index] = true;
                    index += 1;
                }
            }
        }
    }
    readable
}

// ---- The documents --------------------------------------------------------------------------------------------------

/// SplitMix64 of the seed and the document's index and a counter, so document `n` doesn't need the ones before it.
fn hash(seed: u64, index: u64, counter: u64) -> u64 {
    let mut z = seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ counter.wrapping_mul(0xD1B5_4A32_D192_ED03);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Document `index`: a package manifest, a user record, a log event, editor settings, an order or sensor readings,
/// valid JSON of at most 256 ASCII bytes, which of them following from a hash of the index.
pub fn document(seed: u64, index: u64) -> String {
    let mut counter = 0;
    let mut next = |bound: u64| {
        counter += 1;
        hash(seed, index, counter) % bound
    };
    const NAMES: [&str; 8] = [
        "Ada Lovelace",
        "Alan Turing",
        "Grace Hopper",
        "Edsger Dijkstra",
        "Barbara Liskov",
        "Ken Thompson",
        "Margaret Hamilton",
        "Donald Knuth",
    ];
    const WORDS: [&str; 12] =
        ["the", "an", "a", "error", "timeout", "cache", "request", "user", "disk", "connection", "reset", "retry"];
    const LEVELS: [&str; 4] = ["debug", "info", "warn", "error"];
    let date = format!("20{:02}-{:02}-{:02}", 20 + next(7), 1 + next(12), 1 + next(28));
    let mut text = match next(6) {
        0 => format!(
            r#"{{"name": "{}-{}", "version": "{}.{}.{}", "private": {}, "scripts": {{"test": "bun test", "build": "tsc"}}}}"#,
            WORDS[next(12) as usize],
            WORDS[next(12) as usize],
            next(4),
            next(20),
            next(10),
            next(2) == 0
        ),
        1 => format!(
            r#"{{"id": {}, "name": "{}", "email": "user{}@example.com", "joined": "{date}", "admin": {}}}"#,
            next(100_000),
            NAMES[next(8) as usize],
            next(1000),
            next(2) == 0
        ),
        2 => format!(
            r#"{{"time": "{date}T{:02}:{:02}:{:02}Z", "level": "{}", "message": "{} {} {}", "attempt": {}}}"#,
            next(24),
            next(60),
            next(60),
            LEVELS[next(4) as usize],
            WORDS[next(12) as usize],
            WORDS[next(12) as usize],
            WORDS[next(12) as usize],
            next(5)
        ),
        3 => format!(
            "{{\n\t\"editor.fontSize\": {},\n\t\"editor.tabSize\": {},\n\t\"files.autoSave\": \"{}\",\n\t\"theme\": \"{}\\tdark\"\n}}",
            10 + next(10),
            2 + 2 * next(2),
            ["off", "afterDelay", "onFocusChange"][next(3) as usize],
            WORDS[next(12) as usize]
        ),
        4 => format!(
            r#"{{"order": {}, "placed": "{date}", "items": [{{"sku": "A-{}", "qty": {}}}, {{"sku": "B-{}", "qty": {}}}], "note": "{} {}"}}"#,
            next(1_000_000),
            next(1000),
            1 + next(9),
            next(1000),
            1 + next(9),
            WORDS[next(12) as usize],
            WORDS[next(12) as usize]
        ),
        _ => format!(
            r#"{{"sensor": "s{}", "readings": [{}.{}, {}.{}, {}.{}], "unit": "C", "ok": {}}}"#,
            next(100),
            next(40),
            next(10),
            next(40),
            next(10),
            next(40),
            next(10),
            next(10) != 0
        ),
    };
    text.truncate(POSITIONS);
    text
}
