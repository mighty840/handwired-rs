//! `#[handwired]`: trace a subset of Rust into an exact f32 circuit.
//!
//! For `fn foo(a: A, b: B) -> R`, the attribute keeps `foo` unchanged, as the
//! oracle, and adds:
//!
//! - `foo_hw(a: <A as HwType>::Val, ...) -> <R as HwType>::Val`: the same body,
//!   where values that depend on the inputs are circuit values. Running it
//!   appends the program to the circuit being built. Other `#[handwired]`
//!   functions call it.
//! - `foo_circuit() -> Compiled<(A, B), R>`: builds and compiles the circuit.
//!
//! Rust can't overload comparisons, `&&`/`||`, shifts, indexing that returns a
//! new value, assignment, `if`, `match` or `as`, so those are rewritten into
//! calls of `handwired_rs::dsl`. A `while` or `loop` declares
//! `#[max_iterations(N)]` and becomes a `for` loop over `0..N` that breaks
//! where it would end. Everything else, e.g. `a + b` or `x.min(y)`, stays as
//! written and resolves to `Val`'s operator impls.

use std::collections::{HashMap, HashSet};

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as Tokens};
use quote::{ToTokens, format_ident, quote, quote_spanned};
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::{
    BinOp, Block, Expr, ExprIf, ExprMatch, FnArg, Ident, ItemFn, Local, Pat, ReturnType, Stmt, Type, UnOp,
    parse_macro_input, parse_quote,
};

mod derive;
mod loops;

/// `#[derive(HwType)]`: a struct with named fields of scalars, arrays and other derived structs as a `#[handwired]`
/// parameter, local or result. It adds `<Name>Hw`, the struct's circuit form, which `#[handwired]` code builds from
/// `Name { ... }` literals. Fields take `#[range(lo, hi)]`, `#[ranges([...])]` and `#[grid(step)]`. The struct also
/// needs `Clone`, `Copy`, `Debug` and `PartialEq` to be a parameter or result.
///
/// On an enum whose variants carry no data it makes the enum a scalar, its discriminant in one cell, for `match`,
/// `matches!`, `if let`, `==` and `as` in `#[handwired]` code. The enum needs `Clone`, `Copy`, `Debug` and
/// `PartialEq`, plus `PartialOrd` for `<` and friends.
///
/// On an enum whose variants carry data it adds `<Name>Hw`: the variant's index in one cell, then every variant's
/// fields, which hold 0 where the value is another variant. `#[handwired]` code builds it with `Name::Variant(..)` and
/// `Name::Variant { .. }`, and matches it with patterns that bind the fields.
#[proc_macro_derive(HwType, attributes(range, ranges, grid))]
pub fn derive_hw_type(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    match derive::expand(input) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

#[proc_macro_attribute]
pub fn handwired(attr: TokenStream, item: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        let attr = Tokens::from(attr);
        return syn::Error::new(attr.span(), "#[handwired] takes no arguments").to_compile_error().into();
    }
    let function = parse_macro_input!(item as ItemFn);
    match expand(function) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

struct Param {
    name: Ident,
    mutable: bool,
    ty: Type,
    range: Option<(Expr, Expr)>,
    grid: Option<Expr>,
    ranges: Option<Expr>,
}

fn expand(mut function: ItemFn) -> syn::Result<Tokens> {
    let signature = &function.sig;
    if !signature.generics.params.is_empty() || signature.generics.where_clause.is_some() {
        return Err(syn::Error::new(signature.generics.span(), "#[handwired] functions can't be generic yet"));
    }
    if let Some(token) = &signature.asyncness {
        return Err(syn::Error::new(token.span(), "#[handwired] functions can't be async"));
    }
    if let Some(token) = &signature.unsafety {
        return Err(syn::Error::new(token.span(), "#[handwired] functions can't be unsafe"));
    }
    let mut params = Vec::new();
    for input in function.sig.inputs.iter_mut() {
        let FnArg::Typed(typed) = input else {
            return Err(syn::Error::new(input.span(), "#[handwired] functions can't take self"));
        };
        let Pat::Ident(pattern) = &*typed.pat else {
            return Err(syn::Error::new(typed.pat.span(), "#[handwired] parameters must be plain names"));
        };
        let mut param = Param {
            name: pattern.ident.clone(),
            mutable: pattern.mutability.is_some(),
            ty: (*typed.ty).clone(),
            range: None,
            grid: None,
            ranges: None,
        };
        // Take #[range], #[ranges] and #[grid] off the parameter, since rustc rejects unknown parameter attributes.
        let mut kept = Vec::new();
        for attribute in typed.attrs.drain(..) {
            if attribute.path().is_ident("range") {
                let bounds =
                    attribute.parse_args_with(syn::punctuated::Punctuated::<Expr, syn::Token![,]>::parse_terminated)?;
                if bounds.len() != 2 {
                    return Err(syn::Error::new(attribute.span(), "expected #[range(lo, hi)]"));
                }
                param.range = Some((bounds[0].clone(), bounds[1].clone()));
            } else if attribute.path().is_ident("ranges") {
                param.ranges = Some(attribute.parse_args::<Expr>()?);
            } else if attribute.path().is_ident("grid") {
                param.grid = Some(attribute.parse_args::<Expr>()?);
            } else {
                kept.push(attribute);
            }
        }
        typed.attrs = kept;
        params.push(param);
    }
    let return_type: Type = match &function.sig.output {
        ReturnType::Default => parse_quote!(()),
        ReturnType::Type(_, ty) => (**ty).clone(),
    };

    // `while` and `loop` with #[max_iterations(N)]: a bounded `for` loop in the circuit's copy, a counted loop in
    // the function, which stays the oracle.
    let mut hw_block = (*function.block).clone();
    if let Some(error) = loops::rewrite(&mut hw_block, false).into_iter().reduce(|mut all, error| {
        all.combine(error);
        all
    }) {
        return Err(error);
    }
    loops::rewrite(&mut function.block, true);

    let mut rewriter = Rewriter::new(return_type.clone());
    for param in &params {
        rewriter.types.insert(param.name.to_string(), param.ty.clone());
    }
    // An early `return` can't leave the trace, which runs every path: it records its value
    // where no return happened yet, and the function's value picks the recorded one.
    rewriter.returns = exits(&hw_block.stmts).returns;
    let body = rewriter.block(&hw_block, true, Some(&return_type), false);
    let body = if rewriter.returns {
        quote! {{
            let mut __hw_done: ::handwired_rs::dsl::Val<bool> = ::handwired_rs::dsl::Conv::conv(false);
            let mut __hw_ret: <#return_type as ::handwired_rs::dsl::HwType>::Val = <#return_type as ::handwired_rs::dsl::HwType>::zero();
            let __hw_tail: <#return_type as ::handwired_rs::dsl::HwType>::Val = #body;
            ::handwired_rs::dsl::Mux::select(__hw_done, __hw_tail, __hw_ret)
        }}
    } else {
        body
    };
    if let Some(error) = rewriter.errors.into_iter().reduce(|mut all, error| {
        all.combine(error);
        all
    }) {
        return Err(error);
    }

    let hw_doc = format!("The circuit form of [`{}`], which `#[handwired]` code calls.", function.sig.ident);
    let circuit_doc =
        format!("Builds and compiles the circuit of [`{}`], with the function kept as its oracle.", function.sig.ident);
    let lint_attributes: Vec<&syn::Attribute> = function
        .attrs
        .iter()
        .filter(|attribute| ["allow", "expect", "warn", "deny"].iter().any(|lint| attribute.path().is_ident(lint)))
        .collect();
    // `_circuit()` takes no parameters, so an `#[expect]` about them would go unfulfilled there.
    let circuit_lint_attributes: Vec<&syn::Attribute> =
        lint_attributes.iter().copied().filter(|attribute| !attribute.path().is_ident("expect")).collect();
    let visibility = &function.vis;
    let name = &function.sig.ident;
    let hw_name = format_ident!("{}_hw", name);
    let circuit_name = format_ident!("{}_circuit", name);
    let names: Vec<&Ident> = params.iter().map(|param| &param.name).collect();
    let types: Vec<&Type> = params.iter().map(|param| &param.ty).collect();
    let hw_params = params.iter().map(|param| {
        let (name, ty) = (&param.name, &param.ty);
        let mutability = param.mutable.then(|| quote!(mut));
        quote!(#mutability #name: <#ty as ::handwired_rs::dsl::HwType>::Val)
    });
    let attrs = params.iter().map(|param| {
        let range = match &param.range {
            Some((low, high)) => quote!(::core::option::Option::Some(((#low) as f64, (#high) as f64))),
            None => quote!(::core::option::Option::None),
        };
        let grid = match &param.grid {
            Some(step) => quote!(::core::option::Option::Some((#step) as f64)),
            None => quote!(::core::option::Option::None),
        };
        let ranges = match &param.ranges {
            // A list written out, e.g. #[ranges([(2, 20), (1, 200)])]: each bound becomes an f64.
            Some(Expr::Array(list)) => {
                let mut pairs = Vec::new();
                for element in &list.elems {
                    match element {
                        Expr::Tuple(pair) if pair.elems.len() == 2 => {
                            let (low, high) = (&pair.elems[0], &pair.elems[1]);
                            pairs.push(quote!(((#low) as f64, (#high) as f64)));
                        }
                        other => {
                            return syn::Error::new(other.span(), "expected a (lo, hi) pair").to_compile_error();
                        }
                    }
                }
                quote!(::core::option::Option::Some(&[#(#pairs),*]))
            }
            // A constant of type [(f64, f64); N].
            Some(other) => quote!(::core::option::Option::Some(&#other)),
            None => quote!(::core::option::Option::None),
        };
        if param.range.is_some() && param.ranges.is_some() {
            return syn::Error::new(param.name.span(), "use either #[range] or #[ranges], not both").to_compile_error();
        }
        quote!(::handwired_rs::dsl::Attrs::new(#range, #grid, #ranges))
    });
    let declarations = params.iter().enumerate().map(|(index, param)| {
        let (name, ty) = (&param.name, &param.ty);
        quote!(let #name = <#ty as ::handwired_rs::dsl::HwType>::declare(__inputs, &__ATTRS[#index]);)
    });

    Ok(quote! {
        #function

        #[allow(unused_mut, unused_variables, unused_assignments, unused_parens, unused_braces, unreachable_code)]
        #[allow(clippy::all)]
        #(#lint_attributes)*
        #[doc = #hw_doc]
        #visibility fn #hw_name(#(#hw_params),*) -> <#return_type as ::handwired_rs::dsl::HwType>::Val #body

        #[allow(clippy::all)]
        #(#circuit_lint_attributes)*
        #[doc = #circuit_doc]
        #visibility fn #circuit_name() -> ::handwired_rs::dsl::Compiled<(#(#types,)*), #return_type> {
            const __ATTRS: &[::handwired_rs::dsl::Attrs] = &[#(#attrs),*];
            ::handwired_rs::dsl::Compiled::build(
                |(#(#names,)*): (#(#types,)*)| #name(#(#names),*),
                __ATTRS,
                |__inputs| {
                    #(#declarations)*
                    #hw_name(#(#names),*)
                },
            )
        }
    })
}

struct Rewriter {
    return_type: Type,
    /// Declared types of names, from parameters and annotated `let`s.
    types: HashMap<String, Type>,
    errors: Vec<syn::Error>,
    /// Set while rewriting a branch whose condition depends on the inputs, to
    /// the loop depth there: both branches run, so `break` or `continue` aimed
    /// at a loop at or above it sets that loop's flags instead of leaving it.
    dynamic_loop_floor: Option<usize>,
    loop_depth: usize,
    /// For each enclosing loop, its depth when it has break/continue flags.
    flagged_loops: Vec<Option<usize>>,
    /// Whether the function returns early anywhere, so `return` records into flags.
    returns: bool,
    counter: usize,
}

impl Rewriter {
    fn new(return_type: Type) -> Rewriter {
        Rewriter {
            return_type,
            types: HashMap::new(),
            errors: Vec::new(),
            dynamic_loop_floor: None,
            loop_depth: 0,
            flagged_loops: Vec::new(),
            returns: false,
            counter: 0,
        }
    }

    /// The flag that says the innermost flagged loop stopped running, on the
    /// paths where it did: assignments inside the loop don't apply there.
    fn halted(&self) -> Option<Ident> {
        self.flagged_loops.iter().rev().flatten().next().map(|&depth| halted_flag(depth))
    }

    fn placeholder(expected: Option<&Type>) -> Tokens {
        match expected {
            Some(Type::Infer(_)) => quote!(::handwired_rs::dsl::placeholder()),
            Some(ty) => quote!(::handwired_rs::dsl::placeholder::<<#ty as ::handwired_rs::dsl::HwType>::Val>()),
            None => quote!(::handwired_rs::dsl::placeholder()),
        }
    }

    fn error(&mut self, span: Span, message: &str) -> Tokens {
        self.errors.push(syn::Error::new(span, message));
        quote!(())
    }

    fn fresh(&mut self, base: &str) -> Ident {
        self.counter += 1;
        format_ident!("__{}{}", base, self.counter)
    }

    /// `expr` rewritten as a value whose circuit form is `expected`'s. Where `expected` is `_`, e.g. for a struct
    /// field, whose type the macro can't name, rustc infers it from where the value goes; an `if`, `match`, block,
    /// tuple or array then converts each of its parts, since converting what they make again would leave its type
    /// open.
    fn converted(&mut self, expr: &Expr, expected: Option<&Type>) -> Tokens {
        let rewritten = self.expr(expr, true, expected);
        if is_inferred(expected) && converts_parts(expr) {
            rewritten
        } else {
            wrap_expected(rewritten, expected, expr.span())
        }
    }

    /// The type an expression is known to have, for a plain variable.
    fn known_type(&self, expr: &Expr) -> Option<Type> {
        match expr {
            Expr::Path(path) => path.path.get_ident().and_then(|ident| self.types.get(&ident.to_string()).cloned()),
            Expr::Paren(inner) => self.known_type(&inner.expr),
            Expr::Index(index) => match self.known_type(&index.expr)? {
                Type::Array(array) => Some(*array.elem),
                _ => None,
            },
            Expr::Field(field) => match (self.known_type(&field.base)?, &field.member) {
                (Type::Tuple(tuple), syn::Member::Unnamed(index)) => {
                    tuple.elems.iter().nth(index.index as usize).cloned()
                }
                _ => None,
            },
            _ => None,
        }
    }

    // ---- Blocks and statements ----------------------------------------------------------------------------------------

    /// A block. When `value` is set, its tail is its value, converted to the
    /// circuit form of `expected` when that's known, or lifted to a circuit
    /// value when `lift` is set. A tail that leaves early (`return`, `break`)
    /// still needs a value of the right type, since every path is traced.
    fn block(&mut self, block: &Block, value: bool, expected: Option<&Type>, lift: bool) -> Tokens {
        let saved_types = self.types.clone();
        let mut statements = Vec::new();
        let count = block.stmts.len();
        for (index, statement) in block.stmts.iter().enumerate() {
            let is_tail = index + 1 == count;
            // A macro in statement position, e.g. a tail `matches!(..)`, is handled as the expression it is.
            let as_expression;
            let statement = match statement {
                Stmt::Macro(mac) => {
                    let expr = Expr::Macro(syn::ExprMacro { attrs: mac.attrs.clone(), mac: mac.mac.clone() });
                    as_expression = Stmt::Expr(expr, mac.semi_token);
                    &as_expression
                }
                statement => statement,
            };
            statements.push(match statement {
                Stmt::Local(local) => self.local(local),
                Stmt::Expr(expr, None) if is_tail && value && diverges(expr) => {
                    let rewritten = self.expr(expr, false, None);
                    let placeholder = Rewriter::placeholder(expected);
                    quote!(#rewritten; #placeholder)
                }
                Stmt::Expr(expr, None) if is_tail && value => {
                    let rewritten = self.converted(expr, expected);
                    if lift && expected.is_none() {
                        quote_spanned!(expr.span()=> ::handwired_rs::dsl::Lift::lift(#rewritten))
                    } else {
                        rewritten
                    }
                }
                Stmt::Expr(expr, semicolon) => {
                    let rewritten = self.expr(expr, false, None);
                    quote!(#rewritten #semicolon)
                }
                Stmt::Item(item) => self.error(item.span(), "#[handwired] functions can't declare items inside"),
                Stmt::Macro(mac) => self.error(mac.span(), "macros aren't supported inside #[handwired] functions"),
            });
        }
        if value && matches!(block.stmts.last(), Some(Stmt::Expr(expr, Some(_))) if diverges(expr)) {
            statements.push(Rewriter::placeholder(expected));
        }
        self.types = saved_types;
        quote!({ #(#statements)* })
    }

    fn local(&mut self, local: &Local) -> Tokens {
        let span = local.span();
        if let Some(init) = &local.init
            && let Some((_, diverge)) = &init.diverge
        {
            return self.error(diverge.span(), "let-else isn't supported in #[handwired] functions");
        }
        let (pattern, annotated) = match &local.pat {
            Pat::Type(typed) => (&*typed.pat, Some((*typed.ty).clone())),
            other => (other, None),
        };
        let init = local.init.as_ref().map(|init| self.expr(&init.expr, true, annotated.as_ref()));
        if let (Pat::Ident(ident), Some(ty)) = (pattern, &annotated) {
            self.types.insert(ident.ident.to_string(), ty.clone());
        } else if let Pat::Ident(ident) = pattern {
            self.types.remove(&ident.ident.to_string());
        }
        match (annotated, init) {
            (Some(ty), Some(init)) => quote_spanned!(span=>
                let #pattern: <#ty as ::handwired_rs::dsl::HwType>::Val = ::handwired_rs::dsl::Conv::<<#ty as ::handwired_rs::dsl::HwType>::Val>::conv(#init);
            ),
            (Some(ty), None) => {
                // Branches that assign it are merged, which reads it first; rustc already proved the
                // function never reads it before an assignment, so any start value is fine.
                let name = match pattern {
                    Pat::Ident(ident) => &ident.ident,
                    _ => return self.error(span, "a `let` without a value must bind a plain name"),
                };
                quote_spanned!(span=> let mut #name: <#ty as ::handwired_rs::dsl::HwType>::Val = <#ty as ::handwired_rs::dsl::HwType>::zero();)
            }
            (None, Some(init)) if matches!(pattern, Pat::Tuple(tuple) if tuple.elems.iter().all(|element| matches!(element, Pat::Ident(ident) if ident.subpat.is_none() && ident.by_ref.is_none()) || matches!(element, Pat::Wild(_)))) =>
            {
                // `let (mut a, b) = ..`: bind each name on its own, so each `mut` one starts as a circuit value.
                let Pat::Tuple(tuple) = pattern else { unreachable!() };
                let whole = self.fresh("tuple");
                let mut bindings = vec![quote_spanned!(span=> let #whole = #init;)];
                for (index, element) in tuple.elems.iter().enumerate() {
                    let Pat::Ident(ident) = element else { continue };
                    self.types.remove(&ident.ident.to_string());
                    let index = syn::Index::from(index);
                    bindings.push(if ident.mutability.is_some() {
                        quote_spanned!(span=> let #ident = ::handwired_rs::dsl::Lift::lift(#whole.#index);)
                    } else {
                        quote_spanned!(span=> let #ident = #whole.#index;)
                    });
                }
                quote!(#(#bindings)*)
            }
            (None, Some(init)) => {
                // A mutable binding may take circuit values later, so it starts as one.
                let mutable = matches!(pattern, Pat::Ident(ident) if ident.mutability.is_some());
                if mutable {
                    quote_spanned!(span=> let #pattern = ::handwired_rs::dsl::Lift::lift(#init);)
                } else {
                    quote_spanned!(span=> let #pattern = #init;)
                }
            }
            (None, None) => quote_spanned!(span=> let #pattern;),
        }
    }

    // ---- Expressions --------------------------------------------------------------------------------------------------

    /// Rewrite an expression. `value` says whether its result is used, which
    /// decides how an `if` is translated; `expected` is its type when known.
    fn expr(&mut self, expr: &Expr, value: bool, expected: Option<&Type>) -> Tokens {
        let span = expr.span();
        match expr {
            Expr::Lit(_) | Expr::Path(_) => expr.to_token_stream(),
            Expr::Paren(inner) => {
                let rewritten = self.expr(&inner.expr, value, expected);
                quote_spanned!(span=> (#rewritten))
            }
            Expr::Group(inner) => self.expr(&inner.expr, value, expected),
            Expr::Tuple(tuple) => {
                let element_types: Vec<Option<Type>> = match expected {
                    Some(Type::Tuple(types)) if types.elems.len() == tuple.elems.len() => {
                        types.elems.iter().cloned().map(Some).collect()
                    }
                    Some(Type::Infer(_)) => vec![Some(inferred()); tuple.elems.len()],
                    _ => vec![None; tuple.elems.len()],
                };
                let elements: Vec<Tokens> = tuple
                    .elems
                    .iter()
                    .zip(&element_types)
                    .map(|(element, ty)| self.converted(element, ty.as_ref()))
                    .collect();
                if elements.len() == 1 {
                    let element = &elements[0];
                    quote_spanned!(span=> (#element,))
                } else {
                    quote_spanned!(span=> (#(#elements),*))
                }
            }
            Expr::Array(array) => {
                let element_type = match expected {
                    Some(Type::Array(types)) => Some((*types.elem).clone()),
                    Some(Type::Infer(_)) => Some(inferred()),
                    _ => None,
                };
                let elements: Vec<Tokens> =
                    array.elems.iter().map(|element| self.converted(element, element_type.as_ref())).collect();
                quote_spanned!(span=> [#(#elements),*])
            }
            Expr::Repeat(repeat) => {
                let element_type = match expected {
                    Some(Type::Array(types)) => Some((*types.elem).clone()),
                    Some(Type::Infer(_)) => Some(inferred()),
                    _ => None,
                };
                let element = self.converted(&repeat.expr, element_type.as_ref());
                let length = &repeat.len;
                quote_spanned!(span=> [#element; #length])
            }
            Expr::Unary(unary) => {
                let operand = self.expr(&unary.expr, true, None);
                match unary.op {
                    UnOp::Not(_) => quote_spanned!(span=> !#operand),
                    UnOp::Neg(_) => quote_spanned!(span=> -#operand),
                    _ => self.error(span, "dereferencing isn't supported in #[handwired] functions"),
                }
            }
            Expr::Binary(binary) => self.binary(binary),
            Expr::Assign(assign) => {
                let expected = self.known_type(&assign.left);
                let value = self.expr(&assign.right, true, expected.as_ref());
                self.assign(&assign.left, value)
            }
            Expr::Index(index) => {
                let base = self.expr(&index.expr, true, None);
                let position = self.expr(&index.index, true, None);
                quote_spanned!(span=> ::handwired_rs::dsl::index(&#base, #position))
            }
            Expr::Field(field) => {
                let base = self.expr(&field.base, true, None);
                let member = &field.member;
                quote_spanned!(span=> #base.#member)
            }
            Expr::If(ExprIf { cond, then_branch, else_branch, .. }) if matches!(**cond, Expr::Let(_)) => {
                // `if let p = e { a } else { b }` is `match e { p => { a } _ => { b } }`.
                let Expr::Let(binding) = &**cond else { unreachable!() };
                let (pattern, scrutinee) = (&binding.pat, &binding.expr);
                let otherwise: Expr = match else_branch {
                    Some((_, otherwise)) => (**otherwise).clone(),
                    None => parse_quote!({}),
                };
                let lowered: ExprMatch = parse_quote!(match #scrutinee { #pattern => #then_branch, _ => #otherwise });
                self.match_expr(&lowered, value, expected)
            }
            Expr::If(if_expr) => self.if_expr(if_expr, value, expected),
            Expr::Match(match_expr) => self.match_expr(match_expr, value, expected),
            Expr::Block(block) => {
                if block.label.is_some() {
                    return self.error(span, "labeled blocks aren't supported in #[handwired] functions");
                }
                self.block(&block.block, value, expected, false)
            }
            Expr::MethodCall(call) => {
                let receiver = self.expr(&call.receiver, true, None);
                let args: Vec<Tokens> = call.args.iter().map(|arg| self.expr(arg, true, None)).collect();
                let (method, turbofish) = (&call.method, &call.turbofish);
                quote_spanned!(span=> #receiver.#method #turbofish(#(#args),*))
            }
            Expr::Call(call) => {
                let lowercase_call = |rewriter: &mut Rewriter| -> Vec<Tokens> {
                    call.args.iter().map(|arg| rewriter.expr(arg, true, None)).collect()
                };
                match &*call.func {
                    // `Shape::Rect(w, h)` or `Some(x)`: a variant with its fields.
                    Expr::Path(path)
                        if path.qself.is_none()
                            && path.path.segments.last().is_some_and(|last| {
                                last.ident.to_string().starts_with(|first: char| first.is_ascii_uppercase())
                            }) =>
                    {
                        let fields: Vec<(syn::Member, &Expr)> = call
                            .args
                            .iter()
                            .enumerate()
                            .map(|(index, arg)| (syn::Member::Unnamed(syn::Index::from(index)), arg))
                            .collect();
                        self.variant(&path.path, fields, expected, span)
                    }
                    // A call to another #[handwired] function goes to its circuit form.
                    Expr::Path(path)
                        if path.qself.is_none()
                            && path.path.get_ident().is_some_and(|ident| {
                                ident.to_string().starts_with(|first: char| first.is_ascii_lowercase() || first == '_')
                            }) =>
                    {
                        let args = lowercase_call(self);
                        let hw_name = format_ident!("{}_hw", path.path.get_ident().unwrap());
                        quote_spanned!(span=> #hw_name(#(::handwired_rs::dsl::Conv::conv(#args)),*))
                    }
                    func => {
                        let args = lowercase_call(self);
                        let func = self.expr(func, true, None);
                        quote_spanned!(span=> #func(#(#args),*))
                    }
                }
            }
            Expr::Cast(cast) => {
                let operand = self.expr(&cast.expr, true, None);
                let ty = &cast.ty;
                quote_spanned!(span=> ::handwired_rs::dsl::cast::<#ty, _>(#operand))
            }
            Expr::ForLoop(for_loop) => {
                // The range is native, so the loop unrolls while the circuit is built.
                let loop_exits = exits(&for_loop.body.stmts);
                if loop_exits.labeled || for_loop.label.is_some() {
                    return self.error(span, "loop labels aren't supported in #[handwired] functions yet");
                }
                let flagged = loop_exits.breaks || loop_exits.continues;
                let outer = self.halted();
                self.loop_depth += 1;
                let depth = self.loop_depth;
                self.flagged_loops.push(flagged.then_some(depth));
                let body = self.block(&for_loop.body, false, None, false);
                self.flagged_loops.pop();
                self.loop_depth -= 1;
                let (pattern, iterable) = (&for_loop.pat, &for_loop.expr);
                if !flagged {
                    return quote_spanned!(span=> for #pattern in #iterable #body);
                }
                // `broken` holds on paths that left the loop; `halted` also on paths that
                // continued past the rest of this iteration, or whose outer loop halted.
                let (broken, halted) = (broken_flag(depth), halted_flag(depth));
                let start = match outer {
                    Some(outer) => quote!(::handwired_rs::dsl::or(#broken, || #outer)),
                    None => quote!(#broken),
                };
                quote_spanned!(span=> {
                    let mut #broken: ::handwired_rs::dsl::Val<bool> = ::handwired_rs::dsl::Conv::conv(false);
                    for #pattern in #iterable {
                        let mut #halted: ::handwired_rs::dsl::Val<bool> = #start;
                        #body
                    }
                })
            }
            Expr::While(_) | Expr::Loop(_) => self.error(
                span,
                "a #[handwired] program runs a fixed number of steps, so loops must be `for` loops over ranges that \
                 don't depend on the inputs",
            ),
            Expr::Return(ret) => {
                let return_type = self.return_type.clone();
                let value = match &ret.expr {
                    Some(value) => self.expr(value, true, Some(&return_type)),
                    None => quote!(()),
                };
                if self.dynamic_loop_floor.is_none() && self.halted().is_none() {
                    // Every path still in the trace reaches this return, so leave for real, like
                    // Rust, keeping the value of paths that already returned under an input condition.
                    return quote_spanned!(span=> return ::handwired_rs::dsl::Mux::select(
                        __hw_done,
                        ::handwired_rs::dsl::Conv::<<#return_type as ::handwired_rs::dsl::HwType>::Val>::conv(#value),
                        __hw_ret,
                    ));
                }
                // Record the value on paths that haven't returned and whose loops still run.
                let running = match self.halted() {
                    Some(halted) => quote!(!#halted),
                    None => quote!(true),
                };
                quote_spanned!(span=> {
                    let __hw_value = ::handwired_rs::dsl::Conv::<<#return_type as ::handwired_rs::dsl::HwType>::Val>::conv(#value);
                    let __hw_take = ::handwired_rs::dsl::and(!__hw_done, || #running);
                    __hw_ret = ::handwired_rs::dsl::Mux::select(__hw_take, __hw_ret, __hw_value);
                    __hw_done = ::handwired_rs::dsl::or(__hw_done, || __hw_take);
                })
            }
            Expr::Break(exit) if exit.label.is_some() || exit.expr.is_some() => {
                self.error(span, "labeled breaks and breaks with values aren't supported in #[handwired] functions")
            }
            Expr::Continue(exit) if exit.label.is_some() => {
                self.error(span, "labeled continues aren't supported in #[handwired] functions")
            }
            Expr::Break(_) | Expr::Continue(_) => {
                let leaves_dynamic_branch = self.dynamic_loop_floor.is_some_and(|floor| self.loop_depth <= floor);
                if !leaves_dynamic_branch {
                    // Under native conditions, leave the unrolled loop for real.
                    return expr.to_token_stream();
                }
                let (broken, halted) = (broken_flag(self.loop_depth), halted_flag(self.loop_depth));
                if matches!(expr, Expr::Break(_)) {
                    quote_spanned!(span=> {
                        #broken = ::handwired_rs::dsl::or(#broken, || !#halted);
                        #halted = ::handwired_rs::dsl::Conv::conv(true);
                    })
                } else {
                    quote_spanned!(span=> { #halted = ::handwired_rs::dsl::Conv::conv(true); })
                }
            }
            Expr::Range(_) => expr.to_token_stream(),
            Expr::Reference(reference) => {
                let operand = self.expr(&reference.expr, true, None);
                let mutability = &reference.mutability;
                quote_spanned!(span=> &#mutability #operand)
            }
            Expr::Closure(_) => self.error(span, "closures aren't supported in #[handwired] functions"),
            Expr::Macro(mac) if mac.mac.path.is_ident("matches") => {
                // `matches!(e, p if g)` is `match e { p if g => true, _ => false }`.
                let parsed = mac.mac.parse_body_with(|input: syn::parse::ParseStream| {
                    let scrutinee: Expr = input.parse()?;
                    input.parse::<syn::Token![,]>()?;
                    let pattern = Pat::parse_multi_with_leading_vert(input)?;
                    let guard = if input.peek(syn::Token![if]) {
                        input.parse::<syn::Token![if]>()?;
                        Some(input.parse::<Expr>()?)
                    } else {
                        None
                    };
                    let _ = input.parse::<Option<syn::Token![,]>>()?;
                    Ok((scrutinee, pattern, guard))
                });
                let (scrutinee, pattern, guard) = match parsed {
                    Ok(parsed) => parsed,
                    Err(error) => {
                        self.errors.push(error);
                        return quote!(());
                    }
                };
                let guard = guard.map(|guard| quote!(if #guard));
                let lowered: ExprMatch = parse_quote!(match #scrutinee { #pattern #guard => true, _ => false });
                self.match_expr(&lowered, value, expected)
            }
            Expr::Macro(_) => self.error(span, "macros aren't supported inside #[handwired] functions"),
            Expr::Let(_) => {
                self.error(span, "`let` is only supported directly in an `if let` in #[handwired] functions")
            }
            Expr::Struct(structure) => {
                // `State { .. }` builds the circuit form, `StateHw`, which #[derive(HwType)] adds.
                if structure.qself.is_some() {
                    return self.error(span, "qualified struct paths aren't supported in #[handwired] functions");
                }
                let segments = &structure.path.segments;
                if segments.len() >= 2
                    && segments[segments.len() - 2]
                        .ident
                        .to_string()
                        .starts_with(|first: char| first.is_ascii_uppercase())
                {
                    // `Shape::Circle { r }`: a variant with named fields.
                    if structure.rest.is_some() {
                        return self
                            .error(span, "`..` isn't supported in a variant's fields in #[handwired] functions");
                    }
                    let fields: Vec<(syn::Member, &Expr)> =
                        structure.fields.iter().map(|field| (field.member.clone(), &field.expr)).collect();
                    return self.variant(&structure.path, fields, expected, span);
                }
                let mut path = structure.path.clone();
                let last = path.segments.last_mut().unwrap();
                last.ident = format_ident!("{}Hw", last.ident);
                let fields: Vec<Tokens> = structure
                    .fields
                    .iter()
                    .map(|field| {
                        let member = &field.member;
                        let value = self.converted(&field.expr, Some(&inferred()));
                        quote_spanned!(field.span()=> #member: #value)
                    })
                    .collect();
                let rest = structure.rest.as_ref().map(|rest| {
                    let rest = self.expr(rest, true, None);
                    quote!(..::handwired_rs::dsl::Lift::lift(#rest))
                });
                quote_spanned!(span=> #path { #(#fields,)* #rest })
            }
            _ => self.error(span, "this expression isn't supported in #[handwired] functions"),
        }
    }

    /// A variant of an enum that carries data, from its path and its fields: the circuit form's value with the
    /// variant's tag, each field set, and the other variants' fields 0. Each field's type comes from the enum, so an
    /// `if` of plain literals there takes it too. `Some`, `Ok` and `Err` take their type from where the value goes when
    /// that's known, else from the value.
    fn variant(
        &mut self,
        path: &syn::Path,
        fields: Vec<(syn::Member, &Expr)>,
        expected: Option<&Type>,
        span: Span,
    ) -> Tokens {
        let last = path.segments.last().unwrap();
        let variant = &last.ident;
        let name = variant.to_string();
        let hw_type = if path.segments.len() == 1 && matches!(name.as_str(), "Some" | "Ok" | "Err") {
            if fields.len() != 1 {
                return self.error(span, &format!("`{name}` takes one value"));
            }
            match expected {
                Some(Type::Infer(_)) if name == "Some" => {
                    // Built whole, as `dsl::some` builds it, without the 0 that `__hw_new_Some` makes first.
                    let value = self.converted(fields[0].1, Some(&inferred()));
                    return quote_spanned!(span=> ::handwired_rs::dsl::OptionHw {
                        __Some__0: #value,
                        __tag: ::handwired_rs::dsl::Val::constant(true),
                    });
                }
                Some(Type::Infer(_)) => quote!(::handwired_rs::dsl::ResultHw<_, _>),
                Some(ty) => quote!(<#ty as ::handwired_rs::dsl::HwType>::Val),
                None => {
                    // Nothing says the type, so it comes from the value (and for `Ok` and `Err`, the other side's
                    // type from where the result goes later).
                    let value = self.expr(fields[0].1, true, None);
                    let helper = format_ident!("{}", name.to_lowercase());
                    return quote_spanned!(span=> ::handwired_rs::dsl::#helper(::handwired_rs::dsl::Lift::lift(#value)));
                }
            }
        } else if path.segments.len() < 2 {
            return self.error(
                span,
                "name a variant with data by its enum in #[handwired] functions, e.g. `Shape::Rect(w, h)`, or this is \
                 a tuple struct, which isn't supported",
            );
        } else {
            let mut enum_path = path.clone();
            enum_path.segments.pop();
            let enum_path = enum_path.segments.into_iter().collect::<syn::punctuated::Punctuated<_, syn::Token![::]>>();
            let leading = &path.leading_colon;
            quote!(<#leading #enum_path as ::handwired_rs::dsl::HwType>::Val)
        };
        let value = self.fresh("variant");
        let new = format_ident!("__hw_new_{}", variant);
        let assignments: Vec<Tokens> = fields
            .into_iter()
            .map(|(member, field)| {
                let name = match member {
                    syn::Member::Named(name) => format_ident!("__{}__{}", variant, name),
                    syn::Member::Unnamed(index) => format_ident!("__{}__{}", variant, index.index),
                };
                let field = self.converted(field, Some(&inferred()));
                quote_spanned!(span=> #value.#name = #field;)
            })
            .collect();
        quote_spanned!(span=> {
            let mut #value = <#hw_type>::#new();
            #(#assignments)*
            #value
        })
    }

    fn binary(&mut self, binary: &syn::ExprBinary) -> Tokens {
        let span = binary.span();
        let compound = match binary.op {
            BinOp::AddAssign(_) => Some(quote!(+)),
            BinOp::SubAssign(_) => Some(quote!(-)),
            BinOp::MulAssign(_) => Some(quote!(*)),
            BinOp::DivAssign(_) => Some(quote!(/)),
            BinOp::RemAssign(_) => Some(quote!(%)),
            BinOp::BitAndAssign(_) => Some(quote!(&)),
            BinOp::BitOrAssign(_) => Some(quote!(|)),
            BinOp::BitXorAssign(_) => Some(quote!(^)),
            BinOp::ShlAssign(_) | BinOp::ShrAssign(_) => {
                // x <<= y becomes x = shl(x, y), since Rust can't overload a shift's amount type here.
                let function = Ident::new(if matches!(binary.op, BinOp::ShlAssign(_)) { "shl" } else { "shr" }, span);
                let current = self.expr(&binary.left, true, None);
                let right = self.expr(&binary.right, true, None);
                let shifted = quote_spanned!(span=> ::handwired_rs::dsl::#function(#current, #right));
                return self.assign(&binary.left, shifted);
            }
            _ => None,
        };
        if let Some(operator) = compound {
            // x op= y becomes x = x op y; the left side reads the current value.
            let current = self.expr(&binary.left, true, None);
            let right = self.expr(&binary.right, true, None);
            let combined = quote_spanned!(span=> #current #operator (#right));
            return self.assign(&binary.left, combined);
        }
        let left = self.expr(&binary.left, true, None);
        let right = self.expr(&binary.right, true, None);
        let call = |function: &str| {
            let function = Ident::new(function, span);
            quote_spanned!(span=> ::handwired_rs::dsl::#function(#left, #right))
        };
        match binary.op {
            BinOp::Lt(_) => call("lt"),
            BinOp::Le(_) => call("le"),
            BinOp::Gt(_) => call("gt"),
            BinOp::Ge(_) => call("ge"),
            BinOp::Eq(_) => call("eq"),
            BinOp::Ne(_) => call("ne"),
            BinOp::And(_) => quote_spanned!(span=> ::handwired_rs::dsl::and(#left, || #right)),
            BinOp::Or(_) => quote_spanned!(span=> ::handwired_rs::dsl::or(#left, || #right)),
            // Two literals stay a native shift, so the literal's type comes from the context, e.g. `b |= 1 << 9`.
            BinOp::Shl(_) | BinOp::Shr(_) if is_literal(&binary.left) && is_literal(&binary.right) => {
                let op = binary.op;
                quote_spanned!(span=> #left #op #right)
            }
            BinOp::Shl(_) => call("shl"),
            BinOp::Shr(_) => call("shr"),
            op => quote_spanned!(span=> #left #op #right),
        }
    }

    /// `target = value`, where value is already rewritten. Inside a loop with
    /// flags, the assignment only applies where the loop still runs.
    fn assign(&mut self, target: &Expr, value: Tokens) -> Tokens {
        let value = match self.halted() {
            Some(halted) => {
                let current = self.expr(target, true, None);
                quote_spanned!(target.span()=> ::handwired_rs::dsl::Mux::select(#halted, ::handwired_rs::dsl::Conv::conv(#value), #current))
            }
            None => value,
        };
        self.assign_to(target, value)
    }

    fn assign_to(&mut self, target: &Expr, value: Tokens) -> Tokens {
        let span = target.span();
        match target {
            // `a[i].field = v` for a struct element: read the element, set the field, and store it back at i.
            Expr::Field(field) if contains_index(&field.base) => {
                let current = self.expr(&field.base, true, None);
                let element = self.fresh("element");
                let member = &field.member;
                let updated = quote_spanned!(span=> {
                    let mut #element = #current;
                    #element.#member = ::handwired_rs::dsl::Conv::conv(#value);
                    #element
                });
                self.assign_to(&field.base, updated)
            }
            Expr::Path(_) | Expr::Field(_) => {
                let target = target.to_token_stream();
                quote_spanned!(span=> #target = ::handwired_rs::dsl::Conv::conv(#value))
            }
            Expr::Paren(inner) => self.assign_to(&inner.expr, value),
            Expr::Index(index) => {
                let position = self.expr(&index.index, true, None);
                match &*index.expr {
                    Expr::Path(_) | Expr::Field(_) => {
                        // Evaluate both before borrowing the array mutably, since the value may read it.
                        let base = index.expr.to_token_stream();
                        let (bound_position, bound_value) = (self.fresh("position"), self.fresh("stored"));
                        quote_spanned!(span=> {
                            let #bound_position = #position;
                            let #bound_value = #value;
                            ::handwired_rs::dsl::store(&mut #base, #bound_position, #bound_value)
                        })
                    }
                    base => {
                        // a[i][j] = v: read a[i], store v at j, then store the row back at i.
                        let row = self.expr(base, true, None);
                        let element = self.fresh("element");
                        let updated = quote_spanned!(span=> {
                            let mut #element = #row;
                            ::handwired_rs::dsl::store(&mut #element, #position, #value);
                            #element
                        });
                        self.assign_to(base, updated)
                    }
                }
            }
            _ => self.error(span, "this assignment target isn't supported in #[handwired] functions"),
        }
    }

    // ---- if and match -------------------------------------------------------------------------------------------------

    fn if_expr(&mut self, if_expr: &ExprIf, value: bool, expected: Option<&Type>) -> Tokens {
        let span = if_expr.span();
        let otherwise = if_expr.else_branch.as_ref().map(|(_, otherwise)| &**otherwise);
        let value = value && otherwise.is_some();
        let mut assigned = assigned_names(&if_expr.then_branch.stmts);
        let mut branch_exits = exits(&if_expr.then_branch.stmts);
        if let Some(otherwise) = otherwise {
            let stmts = vec![Stmt::Expr(otherwise.clone(), None)];
            assigned.extend(assigned_names(&stmts));
            branch_exits = branch_exits.union(exits(&stmts));
        }
        // An exit sets flags, which the branches then merge like any other variable.
        if branch_exits.returns {
            assigned.extend(["__hw_done".to_string(), "__hw_ret".to_string()]);
        }
        if branch_exits.breaks {
            assigned.insert(broken_flag(self.loop_depth).to_string());
        }
        if branch_exits.breaks || branch_exits.continues {
            assigned.insert(halted_flag(self.loop_depth).to_string());
        }
        let exits_early = branch_exits.any();
        let condition = self.expr(&if_expr.cond, true, None);

        if value && assigned.is_empty() && !exits_early {
            // Branches that only compute a value: run one for a native condition, both for a circuit one.
            let then = self.block(&if_expr.then_branch, true, expected, false);
            let otherwise = self.converted(otherwise.unwrap(), expected);
            if is_inferred(expected) {
                // Both branches make the type the value's destination has.
                return quote_spanned!(span=> ::handwired_rs::dsl::choose(#condition, || #then, || #otherwise));
            }
            return quote_spanned!(span=> ::handwired_rs::dsl::if_else(#condition, || #then, || #otherwise));
        }

        // Branches that assign variables or exit early.
        let wrap_value = |tokens: Tokens| -> Tokens {
            if !value {
                tokens
            } else if expected.is_some() {
                wrap_expected(tokens, expected, span)
            } else {
                quote_spanned!(span=> ::handwired_rs::dsl::Lift::lift(#tokens))
            }
        };
        // Values are converted inside the branch blocks, so a branch that exits early can give a
        // placeholder of whatever type the other branch has.
        let else_branch = |rewriter: &mut Rewriter| match otherwise {
            Some(Expr::Block(block)) => rewriter.block(&block.block, value, expected, value),
            Some(otherwise) if value && is_inferred(expected) => rewriter.converted(otherwise, expected),
            Some(otherwise) => {
                let tokens = rewriter.expr(otherwise, value, expected);
                wrap_value(tokens)
            }
            None => quote!({}),
        };
        let static_then = self.block(&if_expr.then_branch, value, expected, value);
        let static_else = else_branch(self);

        let saved_floor = self.dynamic_loop_floor;
        self.dynamic_loop_floor = Some(self.loop_depth);
        let dynamic_then = self.block(&if_expr.then_branch, value, expected, value);
        let dynamic_else = else_branch(self);
        self.dynamic_loop_floor = saved_floor;

        let mut names: Vec<&String> = assigned.iter().collect();
        names.sort();
        let vars: Vec<Ident> = names.iter().map(|name| Ident::new(name, span)).collect();
        let indices: Vec<syn::Index> = (0..vars.len()).map(syn::Index::from).collect();
        let (cond, bit, flag) = (self.fresh("cond"), self.fresh("bit"), self.fresh("static"));
        let (saved, then_vars, then_value, else_vars, else_value) = (
            self.fresh("saved"),
            self.fresh("then_vars"),
            self.fresh("then_value"),
            self.fresh("else_vars"),
            self.fresh("else_value"),
        );
        let branch_value = self.fresh("value");
        quote_spanned!(span=> {
            let #cond = #condition;
            match ::handwired_rs::dsl::Cond::branch(#cond) {
                ::handwired_rs::dsl::Branch::Static(#flag) => if #flag { #static_then } else { #static_else },
                ::handwired_rs::dsl::Branch::Dynamic(#bit) => {
                    let #saved = (#(#vars,)*);
                    let (#then_vars, #then_value) = {
                        let (#(mut #vars,)*) = #saved;
                        let #branch_value = #dynamic_then;
                        ((#(#vars,)*), #branch_value)
                    };
                    let (#else_vars, #else_value) = {
                        let (#(mut #vars,)*) = #saved;
                        let #branch_value = #dynamic_else;
                        ((#(#vars,)*), #branch_value)
                    };
                    #(#vars = ::handwired_rs::dsl::Mux::select(#bit, #else_vars.#indices, #then_vars.#indices);)*
                    ::handwired_rs::dsl::Mux::select(#bit, #else_value, #then_value)
                }
            }
        })
    }

    /// `match` becomes an `if` chain over the scrutinee's value.
    fn match_expr(&mut self, match_expr: &ExprMatch, value: bool, expected: Option<&Type>) -> Tokens {
        let span = match_expr.span();
        let scrutinee_name = self.fresh("scrutinee");
        let scrutinee_expr = self.expr(&match_expr.expr, true, None);
        let subject: Expr = parse_quote!(#scrutinee_name);
        let mut chain: Option<Expr> = None;
        for (index, arm) in match_expr.arms.iter().enumerate().rev() {
            let mut bindings = Vec::new();
            let condition = match pattern_condition(&arm.pat, &subject, &mut bindings) {
                Ok(condition) => condition,
                Err(error) => {
                    self.errors.push(error);
                    return quote!(());
                }
            };
            let body = &arm.body;
            let arm_block: Block = parse_quote!({ #(#bindings)* #body });
            let is_last = index + 1 == match_expr.arms.len();
            chain = Some(if is_last {
                // Rust checked that the arms cover every value, so the last arm is what's left.
                Expr::Block(syn::ExprBlock { attrs: vec![], label: None, block: arm_block })
            } else {
                let condition: Expr = match &arm.guard {
                    Some((_, guard)) => parse_quote!((#condition) && { #(#bindings)* #guard }),
                    None => condition,
                };
                let otherwise = chain.take().unwrap();
                Expr::If(ExprIf {
                    attrs: vec![],
                    if_token: Default::default(),
                    cond: Box::new(condition),
                    then_branch: arm_block,
                    else_branch: Some((Default::default(), Box::new(otherwise))),
                })
            });
        }
        let Some(chain) = chain else {
            return self.error(span, "a match needs at least one arm");
        };
        let chain = self.expr(&chain, value, expected);
        quote_spanned!(span=> { let #scrutinee_name = #scrutinee_expr; #chain })
    }
}

/// `_`, for a value whose type rustc infers from where it goes.
fn inferred() -> Type {
    parse_quote!(_)
}

fn is_inferred(expected: Option<&Type>) -> bool {
    matches!(expected, Some(Type::Infer(_)))
}

/// Whether rewriting `expr` with an expected type converts each of its parts, so that its result already has the
/// expected type.
fn converts_parts(expr: &Expr) -> bool {
    match expr {
        Expr::If(if_expr) => if_expr.else_branch.is_some(),
        Expr::Match(_) | Expr::Block(_) | Expr::Tuple(_) | Expr::Array(_) | Expr::Repeat(_) => true,
        Expr::Paren(inner) => converts_parts(&inner.expr),
        Expr::Group(inner) => converts_parts(&inner.expr),
        _ => false,
    }
}

fn wrap_expected(tokens: Tokens, expected: Option<&Type>, span: Span) -> Tokens {
    match expected {
        Some(Type::Infer(_)) => quote_spanned!(span=> ::handwired_rs::dsl::Conv::conv(#tokens)),
        Some(ty) => {
            quote_spanned!(span=> ::handwired_rs::dsl::Conv::<<#ty as ::handwired_rs::dsl::HwType>::Val>::conv(#tokens))
        }
        None => tokens,
    }
}

/// The condition under which `pattern` matches `subject`, as ordinary Rust
/// that the rewriter then translates, plus the `let`s for its bindings.
fn pattern_condition(pattern: &Pat, subject: &Expr, bindings: &mut Vec<Stmt>) -> syn::Result<Expr> {
    Ok(match pattern {
        Pat::Wild(_) => parse_quote!(true),
        Pat::Lit(literal) => parse_quote!(#subject == #literal),
        Pat::Path(path) => parse_quote!(#subject == #path),
        Pat::Paren(inner) => pattern_condition(&inner.pat, subject, bindings)?,
        Pat::Ident(ident)
            if ident.subpat.is_none()
                && ident.ident.to_string().starts_with(|first: char| first.is_ascii_uppercase()) =>
        {
            // A const, e.g. `MAX => ...`: the oracle compiled, so this names a value rather than binding one.
            let name = &ident.ident;
            parse_quote!(#subject == #name)
        }
        Pat::Ident(ident) => {
            if ident.by_ref.is_some() {
                return Err(syn::Error::new(ident.span(), "`ref` bindings aren't supported"));
            }
            let name = &ident.ident;
            bindings.push(parse_quote!(let #name = #subject;));
            match &ident.subpat {
                Some((_, inner)) => pattern_condition(inner, subject, bindings)?,
                None => parse_quote!(true),
            }
        }
        Pat::Range(range) => {
            let low = range.start.as_ref().map(|low| -> Expr { parse_quote!(#low <= #subject) });
            let high = range.end.as_ref().map(|high| -> Expr {
                match range.limits {
                    syn::RangeLimits::Closed(_) => parse_quote!(#subject <= #high),
                    syn::RangeLimits::HalfOpen(_) => parse_quote!(#subject < #high),
                }
            });
            match (low, high) {
                (Some(low), Some(high)) => parse_quote!((#low) && (#high)),
                (Some(bound), None) | (None, Some(bound)) => bound,
                (None, None) => parse_quote!(true),
            }
        }
        Pat::Or(or) => {
            let mut cases = or.cases.iter();
            let mut condition = pattern_condition(cases.next().unwrap(), subject, bindings)?;
            for case in cases {
                let next = pattern_condition(case, subject, bindings)?;
                condition = parse_quote!((#condition) || (#next));
            }
            condition
        }
        Pat::Tuple(tuple) => {
            let mut condition: Expr = parse_quote!(true);
            for (index, element) in tuple.elems.iter().enumerate() {
                let index = syn::Index::from(index);
                let element_subject: Expr = parse_quote!(#subject.#index);
                let next = pattern_condition(element, &element_subject, bindings)?;
                condition = parse_quote!((#condition) && (#next));
            }
            condition
        }
        Pat::TupleStruct(tuple) => {
            // `Shape::Rect(w, _)` or `Some(x)`: the variant's tag, then each field's own pattern.
            let variant = &tuple.path.segments.last().unwrap().ident;
            let is = format_ident!("__hw_is_{}", variant);
            let mut condition: Expr = parse_quote!(::handwired_rs::dsl::Lift::lift(#subject).#is());
            for (index, element) in tuple.elems.iter().enumerate() {
                if let Pat::Rest(rest) = element {
                    if index + 1 != tuple.elems.len() {
                        return Err(syn::Error::new(rest.span(), "`..` is only supported last in a variant's fields"));
                    }
                    break;
                }
                let field = format_ident!("__{}__{}", variant, index);
                let element_subject: Expr = parse_quote!(::handwired_rs::dsl::Lift::lift(#subject).#field);
                let next = pattern_condition(element, &element_subject, bindings)?;
                condition = parse_quote!((#condition) && (#next));
            }
            condition
        }
        Pat::Struct(structure) => {
            // `Shape::Circle { r, .. }`: the variant's tag, then each named field's pattern.
            if structure.qself.is_some() {
                return Err(syn::Error::new(structure.span(), "qualified paths aren't supported in patterns"));
            }
            let variant = &structure.path.segments.last().unwrap().ident;
            let is = format_ident!("__hw_is_{}", variant);
            let mut condition: Expr = parse_quote!(::handwired_rs::dsl::Lift::lift(#subject).#is());
            for field in &structure.fields {
                let name = match &field.member {
                    syn::Member::Named(name) => format_ident!("__{}__{}", variant, name),
                    syn::Member::Unnamed(index) => format_ident!("__{}__{}", variant, index.index),
                };
                let field_subject: Expr = parse_quote!(::handwired_rs::dsl::Lift::lift(#subject).#name);
                let next = pattern_condition(&field.pat, &field_subject, bindings)?;
                condition = parse_quote!((#condition) && (#next));
            }
            condition
        }
        other => return Err(syn::Error::new(other.span(), "this pattern isn't supported in #[handwired] functions")),
    })
}

/// Whether an expression is an integer literal, possibly negated or parenthesized.
fn is_literal(expr: &Expr) -> bool {
    match expr {
        Expr::Lit(_) => true,
        Expr::Unary(unary) if matches!(unary.op, UnOp::Neg(_)) => is_literal(&unary.expr),
        Expr::Paren(inner) => is_literal(&inner.expr),
        _ => false,
    }
}

/// Whether a place reads through an index, e.g. `units[i].hp`.
fn contains_index(expr: &Expr) -> bool {
    match expr {
        Expr::Index(_) => true,
        Expr::Field(field) => contains_index(&field.base),
        Expr::Paren(inner) => contains_index(&inner.expr),
        _ => false,
    }
}

/// Names that a branch assigns and didn't declare itself: the variables an
/// `if` with a circuit condition has to merge.
fn assigned_names(stmts: &[Stmt]) -> HashSet<String> {
    #[derive(Default)]
    struct Collector {
        assigned: HashSet<String>,
        declared: HashSet<String>,
    }
    fn base_name(expr: &Expr) -> Option<String> {
        match expr {
            Expr::Path(path) => path.path.get_ident().map(|ident| ident.to_string()),
            Expr::Index(index) => base_name(&index.expr),
            Expr::Field(field) => base_name(&field.base),
            Expr::Paren(inner) => base_name(&inner.expr),
            _ => None,
        }
    }
    impl<'ast> Visit<'ast> for Collector {
        fn visit_expr_assign(&mut self, assign: &'ast syn::ExprAssign) {
            self.assigned.extend(base_name(&assign.left));
            syn::visit::visit_expr_assign(self, assign);
        }
        fn visit_expr_binary(&mut self, binary: &'ast syn::ExprBinary) {
            let compound = matches!(
                binary.op,
                BinOp::AddAssign(_)
                    | BinOp::SubAssign(_)
                    | BinOp::MulAssign(_)
                    | BinOp::DivAssign(_)
                    | BinOp::RemAssign(_)
                    | BinOp::BitAndAssign(_)
                    | BinOp::BitOrAssign(_)
                    | BinOp::BitXorAssign(_)
                    | BinOp::ShlAssign(_)
                    | BinOp::ShrAssign(_)
            );
            if compound {
                self.assigned.extend(base_name(&binary.left));
            }
            syn::visit::visit_expr_binary(self, binary);
        }
        fn visit_pat_ident(&mut self, ident: &'ast syn::PatIdent) {
            self.declared.insert(ident.ident.to_string());
            syn::visit::visit_pat_ident(self, ident);
        }
        fn visit_expr_closure(&mut self, _: &'ast syn::ExprClosure) {}
    }
    let mut collector = Collector::default();
    for stmt in stmts {
        collector.visit_stmt(stmt);
    }
    collector.assigned.retain(|name| !collector.declared.contains(name));
    collector.assigned
}

#[derive(Copy, Clone, Default)]
struct Exits {
    returns: bool,
    /// `break`/`continue` aimed at the loop around the scanned code.
    breaks: bool,
    continues: bool,
    labeled: bool,
}

impl Exits {
    fn union(self, other: Exits) -> Exits {
        Exits {
            returns: self.returns || other.returns,
            breaks: self.breaks || other.breaks,
            continues: self.continues || other.continues,
            labeled: self.labeled || other.labeled,
        }
    }

    fn any(self) -> bool {
        self.returns || self.breaks || self.continues || self.labeled
    }
}

fn broken_flag(depth: usize) -> Ident {
    format_ident!("__hw_broken{}", depth)
}

fn halted_flag(depth: usize) -> Ident {
    format_ident!("__hw_halted{}", depth)
}

/// Whether an expression leaves early, so it has no value of its own.
fn diverges(expr: &Expr) -> bool {
    match expr {
        Expr::Return(_) | Expr::Break(_) | Expr::Continue(_) => true,
        Expr::Paren(inner) => diverges(&inner.expr),
        _ => false,
    }
}

/// How code leaves early: `return` anywhere in it, and `break`/`continue`
/// aimed at a loop outside it.
fn exits(stmts: &[Stmt]) -> Exits {
    struct Finder {
        exits: Exits,
        depth: usize,
    }
    impl<'ast> Visit<'ast> for Finder {
        fn visit_expr_return(&mut self, expr: &'ast syn::ExprReturn) {
            self.exits.returns = true;
            syn::visit::visit_expr_return(self, expr);
        }
        fn visit_expr_break(&mut self, expr: &'ast syn::ExprBreak) {
            self.exits.labeled |= expr.label.is_some();
            self.exits.breaks |= self.depth == 0;
        }
        fn visit_expr_continue(&mut self, expr: &'ast syn::ExprContinue) {
            self.exits.labeled |= expr.label.is_some();
            self.exits.continues |= self.depth == 0;
        }
        fn visit_expr_for_loop(&mut self, expr: &'ast syn::ExprForLoop) {
            self.depth += 1;
            syn::visit::visit_expr_for_loop(self, expr);
            self.depth -= 1;
        }
        fn visit_expr_closure(&mut self, _: &'ast syn::ExprClosure) {}
    }
    let mut finder = Finder { exits: Exits::default(), depth: 0 };
    for stmt in stmts {
        finder.visit_stmt(stmt);
    }
    finder.exits
}
