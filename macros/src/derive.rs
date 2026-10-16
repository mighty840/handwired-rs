//! `#[derive(HwType)]`: a struct of scalars, arrays and other derived structs
//! as a `#[handwired]` parameter, local or result.
//!
//! For `struct State { ... }` it adds `StateHw`, the same fields as circuit
//! values, and the traits that let `#[handwired]` code read and assign fields,
//! copy the struct, choose between two of them under an `if`, and index arrays
//! of them. `#[range]`, `#[ranges]` and `#[grid]` on a field declare it.

use proc_macro2::TokenStream as Tokens;
use quote::{format_ident, quote};
use syn::spanned::Spanned;
use syn::{Data, DeriveInput, Expr, Fields};

pub fn expand(input: DeriveInput) -> syn::Result<Tokens> {
    if !input.generics.params.is_empty() || input.generics.where_clause.is_some() {
        return Err(syn::Error::new(input.generics.span(), "#[derive(HwType)] structs can't be generic yet"));
    }
    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new(input.ident.span(), "#[derive(HwType)] works on structs with named fields"));
    };
    let Fields::Named(fields) = &data.fields else {
        return Err(syn::Error::new(input.ident.span(), "#[derive(HwType)] works on structs with named fields"));
    };
    let name = &input.ident;
    let hw_name = format_ident!("{}Hw", name);
    let visibility = &input.vis;
    let dsl = quote!(::handwired_rs::dsl);

    let mut names = Vec::new();
    let mut types = Vec::new();
    let mut visibilities = Vec::new();
    let mut attrs = Vec::new();
    let mut docs = Vec::new();
    for field in &fields.named {
        let doc: Vec<&syn::Attribute> =
            field.attrs.iter().filter(|attribute| attribute.path().is_ident("doc")).collect();
        docs.push(quote!(#(#doc)*));
        names.push(field.ident.clone().unwrap());
        types.push(field.ty.clone());
        visibilities.push(field.vis.clone());
        let (mut range, mut ranges, mut grid) = (quote!(None), quote!(None), quote!(None));
        for attribute in &field.attrs {
            if attribute.path().is_ident("range") {
                let bounds =
                    attribute.parse_args_with(syn::punctuated::Punctuated::<Expr, syn::Token![,]>::parse_terminated)?;
                if bounds.len() != 2 {
                    return Err(syn::Error::new(attribute.span(), "expected #[range(lo, hi)]"));
                }
                let (low, high) = (&bounds[0], &bounds[1]);
                range = quote!(Some(((#low) as f64, (#high) as f64)));
            } else if attribute.path().is_ident("ranges") {
                ranges = match attribute.parse_args::<Expr>()? {
                    Expr::Array(list) => {
                        let mut pairs = Vec::new();
                        for element in &list.elems {
                            let Expr::Tuple(pair) = element else {
                                return Err(syn::Error::new(element.span(), "expected a (lo, hi) pair"));
                            };
                            if pair.elems.len() != 2 {
                                return Err(syn::Error::new(element.span(), "expected a (lo, hi) pair"));
                            }
                            let (low, high) = (&pair.elems[0], &pair.elems[1]);
                            pairs.push(quote!(((#low) as f64, (#high) as f64)));
                        }
                        quote!(Some(&[#(#pairs),*]))
                    }
                    constant => quote!(Some(&#constant)),
                };
            } else if attribute.path().is_ident("grid") {
                let step = attribute.parse_args::<Expr>()?;
                grid = quote!(Some((#step) as f64));
            }
        }
        attrs.push(quote!(#dsl::Attrs { range: #range, grid: #grid, ranges: #ranges }));
    }
    let count = names.len();
    let indices: Vec<usize> = (0..count).collect();
    let doc = format!("The circuit form of [`{name}`], for `#[handwired]` code: the same fields as circuit values.");

    Ok(quote! {
        #[doc = #doc]
        #[derive(Copy, Clone)]
        #[allow(dead_code)]
        #visibility struct #hw_name {
            #(#docs #visibilities #names: <#types as #dsl::HwType>::Val,)*
        }

        // Each field takes the parameter's per-scalar ranges, else its own declarations, else the parameter's.
        #[allow(unused_assignments, unused_variables, unused_mut, clippy::all)]
        const _: () = {
            const FIELD_ATTRS: [#dsl::Attrs; #count] = [#(#attrs),*];

            impl #dsl::HwType for #name {
                type Val = #hw_name;
                const CELLS: usize = 0 #(+ <#types as #dsl::HwType>::CELLS)*;
                fn declare(inputs: &mut #dsl::Inputs, attrs: &#dsl::Attrs) -> #hw_name {
                    attrs.expect_cells(<Self as #dsl::HwType>::CELLS);
                    let mut offset = 0;
                    #hw_name {#(
                        #names: {
                            let cells = <#types as #dsl::HwType>::CELLS;
                            let field = attrs.field(&FIELD_ATTRS[#indices], offset, cells);
                            offset += cells;
                            <#types as #dsl::HwType>::declare(inputs, &field)
                        },
                    )*}
                }
                fn encode(&self, out: &mut Vec<f32>) {
                    #(#dsl::HwType::encode(&self.#names, out);)*
                }
                fn decode(values: &mut dyn Iterator<Item = f32>) -> Self {
                    #name { #(#names: <#types as #dsl::HwType>::decode(values),)* }
                }
                fn outputs(value: &#hw_name, out: &mut Vec<::handwired_rs::Wire>) {
                    #(<#types as #dsl::HwType>::outputs(&value.#names, out);)*
                }
                fn sample(rng: &mut #dsl::Rng, attrs: &#dsl::Attrs) -> Self {
                    attrs.expect_cells(<Self as #dsl::HwType>::CELLS);
                    let mut offset = 0;
                    #name {#(
                        #names: {
                            let cells = <#types as #dsl::HwType>::CELLS;
                            let field = attrs.field(&FIELD_ATTRS[#indices], offset, cells);
                            offset += cells;
                            <#types as #dsl::HwType>::sample(rng, &field)
                        },
                    )*}
                }
                fn zero() -> #hw_name {
                    #hw_name { #(#names: <#types as #dsl::HwType>::zero(),)* }
                }
            }

            impl #dsl::Mux for #hw_name {
                fn select(bit: #dsl::Val<bool>, off: Self, on: Self) -> Self {
                    #hw_name { #(#names: #dsl::Mux::select(bit, off.#names, on.#names),)* }
                }
            }
            impl #dsl::Placeholder for #hw_name {
                fn placeholder() -> Self {
                    #hw_name { #(#names: #dsl::Placeholder::placeholder(),)* }
                }
            }
            impl #dsl::Conv<#hw_name> for #hw_name {
                fn conv(self) -> #hw_name {
                    self
                }
            }
            impl #dsl::Conv<#hw_name> for #name {
                fn conv(self) -> #hw_name {
                    #hw_name { #(#names: #dsl::Conv::conv(self.#names),)* }
                }
            }
            impl #dsl::Lift for #hw_name {
                type Out = #hw_name;
                fn lift(self) -> #hw_name {
                    self
                }
            }
            impl #dsl::Lift for #name {
                type Out = #hw_name;
                fn lift(self) -> #hw_name {
                    #dsl::Conv::conv(self)
                }
            }
            impl #dsl::Unify<#hw_name> for #hw_name {
                type Out = #hw_name;
                fn left(self) -> #hw_name {
                    self
                }
                fn right(other: #hw_name) -> #hw_name {
                    other
                }
            }
            impl #dsl::Unify<#name> for #hw_name {
                type Out = #hw_name;
                fn left(self) -> #hw_name {
                    self
                }
                fn right(other: #name) -> #hw_name {
                    #dsl::Conv::conv(other)
                }
            }
            impl #dsl::Unify<#hw_name> for #name {
                type Out = #hw_name;
                fn left(self) -> #hw_name {
                    #dsl::Conv::conv(self)
                }
                fn right(other: #hw_name) -> #hw_name {
                    other
                }
            }
            impl #dsl::Unify<#name> for #name {
                type Out = #name;
                fn left(self) -> #name {
                    self
                }
                fn right(other: #name) -> #name {
                    other
                }
            }
            impl #dsl::Elem<usize> for #hw_name {
                type Out = #hw_name;
                fn index(items: &[#hw_name], index: usize) -> #hw_name {
                    items[index]
                }
            }
            impl<I: #dsl::IntScalar> #dsl::Elem<#dsl::Val<I>> for #hw_name {
                type Out = #hw_name;
                fn index(items: &[#hw_name], index: #dsl::Val<I>) -> #hw_name {
                    #dsl::Mux::read_dynamic(items, index)
                }
            }
            impl #dsl::Elem<usize> for #name {
                type Out = #name;
                fn index(items: &[#name], index: usize) -> #name {
                    items[index]
                }
            }
            impl<I: #dsl::IntScalar> #dsl::Elem<#dsl::Val<I>> for #name {
                type Out = #hw_name;
                fn index(items: &[#name], index: #dsl::Val<I>) -> #hw_name {
                    let lifted: Vec<#hw_name> = items.iter().map(|&item| #dsl::Conv::conv(item)).collect();
                    #dsl::Mux::read_dynamic(&lifted, index)
                }
            }
        };
    })
}
