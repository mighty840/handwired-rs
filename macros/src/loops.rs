//! `while`, `while let` and `loop` with `#[max_iterations(N)]`.
//!
//! A circuit runs a fixed number of steps, so a loop whose end depends on the
//! inputs declares how many times it may run. The circuit's copy becomes a
//! `for` loop over `0..N` that breaks where the loop would end, which the
//! rewriter already handles: `while c { body }` is
//! `for _ in 0..N { if !(c) { break; } body }`. The function, the oracle,
//! counts its passes and panics with `dsl::iteration_limit` if it would start
//! pass N + 1, which `verify` reports as a failure rather than skipping.

use proc_macro2::TokenStream as Tokens;
use quote::{format_ident, quote_spanned};
use syn::spanned::Spanned;
use syn::visit_mut::VisitMut;
use syn::{Attribute, Block, Expr, parse_quote, parse_quote_spanned};

/// Rewrites every `while`, `while let` and `loop` in a block, for the circuit
/// (`oracle` false) or for the function (`oracle` true).
pub struct Loops {
    pub oracle: bool,
    pub errors: Vec<syn::Error>,
    counter: usize,
}

impl Loops {
    pub fn new(oracle: bool) -> Loops {
        Loops { oracle, errors: Vec::new(), counter: 0 }
    }

    /// Take `#[max_iterations(N)]` off a loop's attributes.
    fn limit(&mut self, attrs: &mut Vec<Attribute>, keyword: &str, span: proc_macro2::Span) -> Option<Expr> {
        let position = attrs.iter().position(|attribute| attribute.path().is_ident("max_iterations"));
        let Some(position) = position else {
            self.errors.push(syn::Error::new(
                span,
                format!(
                    "a `{keyword}` loop in a #[handwired] function needs #[max_iterations(N)]: the circuit runs it N \
                     times, and verify fails on inputs where the function would run it more"
                ),
            ));
            return None;
        };
        match attrs.remove(position).parse_args::<Expr>() {
            Ok(limit) => Some(limit),
            Err(error) => {
                self.errors.push(error);
                None
            }
        }
    }

    /// The function's loop: the same loop, counting its passes.
    fn counted(&mut self, limit: &Expr, body: &mut Block, wrap: impl FnOnce(&Block) -> Expr) -> Expr {
        self.counter += 1;
        let counter = format_ident!("__hw_iterations_{}", self.counter);
        let bound = format_ident!("__HW_MAX_ITERATIONS_{}", self.counter);
        let span = body.span();
        let count: syn::Stmt = parse_quote_spanned!(span=> #counter += 1;);
        let check: syn::Stmt = parse_quote_spanned!(span=>
            if #counter > #bound {
                ::handwired_rs::dsl::iteration_limit(#bound, ::core::concat!(::core::file!(), ":", ::core::line!()));
            }
        );
        body.stmts.splice(0..0, [count, check]);
        let looped = wrap(body);
        // A typed const takes the limit as written, e.g. `64` or `SIZE * 2`, with no cast for clippy to flag.
        parse_quote_spanned!(span=> {
            const #bound: usize = #limit;
            let mut #counter: usize = 0;
            #looped
        })
    }
}

impl VisitMut for Loops {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        // Inner loops first, so an outer loop's body already holds their rewritten forms.
        syn::visit_mut::visit_expr_mut(self, expr);
        let span = expr.span();
        let replacement = match expr {
            Expr::While(looped) => {
                let Some(limit) = self.limit(&mut looped.attrs, "while", span) else { return };
                if looped.label.is_some() {
                    self.errors
                        .push(syn::Error::new(span, "loop labels aren't supported in #[handwired] functions yet"));
                    return;
                }
                if self.oracle {
                    let looped = looped.clone();
                    let mut body = looped.body.clone();
                    self.counted(&limit, &mut body, |body| {
                        let (attrs, cond) = (&looped.attrs, &looped.cond);
                        parse_quote!(#(#attrs)* while #cond #body)
                    })
                } else {
                    let (attrs, body) = (&looped.attrs, &looped.body);
                    let step: Tokens = match &*looped.cond {
                        // `while let p = e { body }` runs body where p matches and stops where it doesn't.
                        Expr::Let(binding) => {
                            let (pattern, scrutinee) = (&binding.pat, &binding.expr);
                            quote_spanned!(span=> if let #pattern = #scrutinee #body else { break; })
                        }
                        cond => quote_spanned!(span=> if !(#cond) { break; } #body),
                    };
                    parse_quote_spanned!(span=> #(#attrs)* for _ in 0..({ const LIMIT: usize = #limit; LIMIT }) { #step })
                }
            }
            Expr::Loop(looped) => {
                let Some(limit) = self.limit(&mut looped.attrs, "loop", span) else { return };
                if looped.label.is_some() {
                    self.errors
                        .push(syn::Error::new(span, "loop labels aren't supported in #[handwired] functions yet"));
                    return;
                }
                if self.oracle {
                    let looped = looped.clone();
                    let mut body = looped.body.clone();
                    self.counted(&limit, &mut body, |body| {
                        let attrs = &looped.attrs;
                        parse_quote!(#(#attrs)* loop #body)
                    })
                } else {
                    let (attrs, body) = (&looped.attrs, &looped.body);
                    parse_quote_spanned!(span=> #(#attrs)* for _ in 0..({ const LIMIT: usize = #limit; LIMIT }) #body)
                }
            }
            _ => return,
        };
        *expr = replacement;
    }

    // Items inside the function are rejected later; don't look into closures' or items' loops here either.
    fn visit_item_mut(&mut self, _: &mut syn::Item) {}
}

/// Rewrite the loops of `block` for the function (`oracle`) or the circuit.
pub fn rewrite(block: &mut Block, oracle: bool) -> Vec<syn::Error> {
    let mut loops = Loops::new(oracle);
    loops.visit_block_mut(block);
    loops.errors
}
