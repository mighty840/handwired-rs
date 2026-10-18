//! `#[derive(HwType)]`: a struct of scalars, arrays and other derived structs
//! as a `#[handwired]` parameter, local or result.
//!
//! For `struct State { ... }` it adds `StateHw`, the same fields as circuit
//! values, and the traits that let `#[handwired]` code read and assign fields,
//! copy the struct, choose between two of them under an `if`, and index arrays
//! of them. `#[range]`, `#[ranges]` and `#[grid]` on a field declare it.
//!
//! For an enum whose variants carry no data it makes the enum a scalar: one
//! cell holding the variant's discriminant, so `==`, `match`, `matches!` and
//! `as` casts work on it, and its circuit form is `Val<Enum>`.
//!
//! For an enum whose variants carry data it adds `EnumHw`: the variant's index
//! as a `Val<u8>`, then every variant's fields, named `__Variant__field`, and
//! per variant `__hw_new_Variant()`, `__hw_is_Variant()` and
//! `__hw_same_Variant()`, which the `#[handwired]` macro calls for
//! constructors, patterns and `==`.

use proc_macro2::TokenStream as Tokens;
use quote::{format_ident, quote};
use syn::spanned::Spanned;
use syn::{Data, DeriveInput, Expr, Fields};

pub fn expand(input: DeriveInput) -> syn::Result<Tokens> {
    if !input.generics.params.is_empty() || input.generics.where_clause.is_some() {
        return Err(syn::Error::new(input.generics.span(), "#[derive(HwType)] structs can't be generic yet"));
    }
    if let Data::Enum(data) = &input.data {
        if data.variants.iter().any(|variant| !matches!(variant.fields, Fields::Unit)) {
            return expand_data_enum(&input, data);
        }
        return expand_enum(&input, data);
    }
    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new(
            input.ident.span(),
            "#[derive(HwType)] works on structs with named fields and enums",
        ));
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
        attrs.push(field_attrs(&field.attrs)?);
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
            impl #dsl::Native for #hw_name {
                type Of = #name;
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

/// A field's `#[range]`, `#[ranges]` and `#[grid]`, as `Attrs`.
fn field_attrs(attributes: &[syn::Attribute]) -> syn::Result<Tokens> {
    let (mut range, mut ranges, mut grid) = (quote!(None), quote!(None), quote!(None));
    for attribute in attributes {
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
    Ok(quote!(::handwired_rs::dsl::Attrs::new(#range, #grid, #ranges)))
}

/// A fieldless enum as a scalar: its discriminant in one cell.
fn expand_enum(input: &DeriveInput, data: &syn::DataEnum) -> syn::Result<Tokens> {
    let name = &input.ident;
    if data.variants.is_empty() {
        return Err(syn::Error::new(name.span(), "#[derive(HwType)] needs an enum with at least one variant"));
    }
    let dsl = quote!(::handwired_rs::dsl);
    let variants: Vec<_> = data.variants.iter().map(|variant| &variant.ident).collect();
    let count = variants.len();
    let label = name.to_string();
    let integers = ["u8", "u16", "u32", "u64", "usize", "i8", "i16", "i32", "i64"].map(|int| format_ident!("{}", int));

    Ok(quote! {
        #[allow(clippy::all)]
        const _: () = {
            const VARIANTS: [#name; #count] = [#(#name::#variants),*];
            const DISCRIMINANTS: [i64; #count] = [#(#name::#variants as i64),*];
            const fn extreme(largest: bool) -> i64 {
                let mut best = DISCRIMINANTS[0];
                let mut index = 1;
                while index < #count {
                    let value = DISCRIMINANTS[index];
                    if (largest && value > best) || (!largest && value < best) {
                        best = value;
                    }
                    index += 1;
                }
                best
            }

            impl #dsl::Scalar for #name {
                const KIND: #dsl::Kind = #dsl::Kind::Int;
                const NAME: &'static str = #label;
                const MIN: f64 = extreme(false) as f64;
                const MAX: f64 = extreme(true) as f64;
                fn to_f64(self) -> f64 {
                    self as i64 as f64
                }
                fn from_f64(value: f64) -> #name {
                    match DISCRIMINANTS.iter().position(|&discriminant| discriminant as f64 == value) {
                        Some(index) => VARIANTS[index],
                        None => panic!("{} has no variant with discriminant {}", #label, value),
                    }
                }
                fn default_value() -> #name {
                    VARIANTS[0]
                }
            }

            impl #dsl::HwType for #name {
                type Val = #dsl::Val<#name>;
                const CELLS: usize = 1;
                fn declare(inputs: &mut #dsl::Inputs, attrs: &#dsl::Attrs) -> #dsl::Val<#name> {
                    #dsl::declare_scalar::<#name>(inputs, attrs)
                }
                fn encode(&self, out: &mut Vec<f32>) {
                    out.push(#dsl::Scalar::to_f64(*self) as f32);
                }
                fn decode(values: &mut dyn Iterator<Item = f32>) -> #name {
                    #dsl::Scalar::from_f64(values.next().expect("too few output values") as f64)
                }
                fn outputs(value: &#dsl::Val<#name>, out: &mut Vec<::handwired_rs::Wire>) {
                    out.push(value.wire());
                }
                fn sample(rng: &mut #dsl::Rng, attrs: &#dsl::Attrs) -> #name {
                    #dsl::sample_variant(rng, attrs, &VARIANTS)
                }
                fn zero() -> #dsl::Val<#name> {
                    #dsl::Val::constant(VARIANTS[0])
                }
            }

            impl #dsl::Conv<#dsl::Val<#name>> for #name {
                fn conv(self) -> #dsl::Val<#name> {
                    #dsl::Val::constant(self)
                }
            }
            impl #dsl::Lift for #name {
                type Out = #dsl::Val<#name>;
                fn lift(self) -> #dsl::Val<#name> {
                    #dsl::Val::constant(self)
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
            impl #dsl::Unify<#dsl::Val<#name>> for #name {
                type Out = #dsl::Val<#name>;
                fn left(self) -> #dsl::Val<#name> {
                    #dsl::Val::constant(self)
                }
                fn right(other: #dsl::Val<#name>) -> #dsl::Val<#name> {
                    other
                }
            }
            impl #dsl::Elem<usize> for #name {
                type Out = #name;
                fn index(items: &[#name], index: usize) -> #name {
                    items[index]
                }
            }
            impl<I: #dsl::IntScalar> #dsl::Elem<#dsl::Val<I>> for #name {
                type Out = #dsl::Val<#name>;
                fn index(items: &[#name], index: #dsl::Val<I>) -> #dsl::Val<#name> {
                    let constants: Vec<#dsl::Val<#name>> = items.iter().map(|&item| #dsl::Val::constant(item)).collect();
                    #dsl::Mux::read_dynamic(&constants, index)
                }
            }
            impl #dsl::Cmp<#name> for #name {
                type Output = bool;
                fn compare(self, relation: ::handwired_rs::Relation, right: #name) -> bool {
                    #dsl::compare_native(self, relation, right)
                }
                fn equal(self, right: #name, equal: bool) -> bool {
                    (self == right) == equal
                }
            }
            impl #dsl::Cmp<#dsl::Val<#name>> for #name {
                type Output = #dsl::Val<bool>;
                fn compare(self, relation: ::handwired_rs::Relation, right: #dsl::Val<#name>) -> #dsl::Val<bool> {
                    #dsl::Cmp::compare(#dsl::Val::constant(self), relation, right)
                }
                fn equal(self, right: #dsl::Val<#name>, equal: bool) -> #dsl::Val<bool> {
                    #dsl::Cmp::equal(#dsl::Val::constant(self), right, equal)
                }
            }
            #(
                impl #dsl::CastTo<#integers> for #name {
                    type Out = #integers;
                    fn cast_to(self) -> #integers {
                        self as #integers
                    }
                }
            )*
        };
    })
}

/// An enum whose variants carry data: the variant's index in one cell, then every variant's fields, each in its own
/// cells, which hold 0 where the value is another variant.
fn expand_data_enum(input: &DeriveInput, data: &syn::DataEnum) -> syn::Result<Tokens> {
    let name = &input.ident;
    let hw_name = format_ident!("{}Hw", name);
    let visibility = &input.vis;
    let dsl = quote!(::handwired_rs::dsl);
    let label = name.to_string();
    if data.variants.len() > 256 {
        return Err(syn::Error::new(name.span(), "#[derive(HwType)] enums with data can have up to 256 variants"));
    }
    let last = (data.variants.len() - 1) as f64;

    // Every field of every variant, in order: its circuit field, type, attributes and offset.
    let (mut hw_fields, mut types, mut attrs) = (Vec::new(), Vec::new(), Vec::new());
    // Per variant: its pattern binding each field, its constructor from values, and its field range.
    struct Variant {
        ident: syn::Ident,
        bind: Tokens,
        build: Tokens,
        fields: std::ops::Range<usize>,
        bindings: Vec<syn::Ident>,
    }
    let mut variants = Vec::new();
    for variant in &data.variants {
        let ident = variant.ident.clone();
        let start = hw_fields.len();
        let mut bindings = Vec::new();
        let mut decoders = Vec::new();
        let mut members = Vec::new();
        for (position, field) in variant.fields.iter().enumerate() {
            let member = match &field.ident {
                Some(field_name) => field_name.to_string(),
                None => position.to_string(),
            };
            hw_fields.push(format_ident!("__{}__{}", ident, member));
            types.push(field.ty.clone());
            attrs.push(field_attrs(&field.attrs)?);
            bindings.push(format_ident!("__field{}", position));
            decoders.push(field.ty.clone());
            members.push(field.ident.clone());
        }
        let (bind, build) = match &variant.fields {
            Fields::Named(_) => {
                let names: Vec<&syn::Ident> = members.iter().map(|member| member.as_ref().unwrap()).collect();
                (
                    quote!(#name::#ident { #(#names: #bindings),* }),
                    quote!(#name::#ident { #(#names: <#decoders as #dsl::HwType>::decode(values)),* }),
                )
            }
            Fields::Unnamed(_) => (
                quote!(#name::#ident(#(#bindings),*)),
                quote!(#name::#ident(#(<#decoders as #dsl::HwType>::decode(values)),*)),
            ),
            Fields::Unit => (quote!(#name::#ident), quote!(#name::#ident)),
        };
        variants.push(Variant { ident, bind, build, fields: start..hw_fields.len(), bindings });
    }
    let count = hw_fields.len();
    let field_indices: Vec<usize> = (0..count).collect();
    // Each field's offset: the tag's cell, then the fields before it.
    let offsets: Vec<Tokens> = (0..count)
        .map(|field| {
            let before = &types[..field];
            quote!(1 #(+ <#before as #dsl::HwType>::CELLS)*)
        })
        .collect();

    let tags: Vec<u8> = (0..variants.len()).map(|index| index as u8).collect();
    let variant_count = variants.len();
    let binds: Vec<&Tokens> = variants.iter().map(|variant| &variant.bind).collect();
    let mut encode_fields = Vec::new();
    let mut decode_variants = Vec::new();
    let mut sample_variants = Vec::new();
    let mut constructors = Vec::new();
    let mut conversions = Vec::new();
    let mut native_equal = Vec::new();
    let mut circuit_equal = Vec::new();
    for (tag, variant) in variants.iter().enumerate() {
        let tag = tag as u8;
        let (ident, bind, build, bindings) = (&variant.ident, &variant.bind, &variant.build, &variant.bindings);
        let fields = &hw_fields[variant.fields.clone()];
        let field_types = &types[variant.fields.clone()];
        let new = format_ident!("__hw_new_{}", ident);
        let is = format_ident!("__hw_is_{}", ident);
        let same = format_ident!("__hw_same_{}", ident);
        if !fields.is_empty() {
            encode_fields.push(quote! {
                match self {
                    #bind => { #(#dsl::HwType::encode(#bindings, out);)* }
                    _ => out.extend(::core::iter::repeat_n(0.0f32, 0 #(+ <#field_types as #dsl::HwType>::CELLS)*)),
                }
            });
        }
        decode_variants.push(quote! {
            if tag == #tag as f32 {
                result = Some(#build);
            } else {
                for _ in 0..(0 #(+ <#field_types as #dsl::HwType>::CELLS)*) {
                    values.next().expect("too few output values");
                }
            }
        });
        let samples = variant.fields.clone().map(|field| {
            let (ty, offset) = (&types[field], &offsets[field]);
            quote!(<#ty as #dsl::HwType>::sample(rng, &attrs.field(&FIELD_ATTRS[#field], #offset, <#ty as #dsl::HwType>::CELLS)))
        });
        let sample_build = match &data.variants[tag as usize].fields {
            Fields::Named(named) => {
                let names = named.named.iter().map(|field| field.ident.as_ref().unwrap());
                quote!(#name::#ident { #(#names: #samples),* })
            }
            Fields::Unnamed(_) => quote!(#name::#ident(#(#samples),*)),
            Fields::Unit => quote!(#name::#ident),
        };
        sample_variants.push(quote!(#tag => #sample_build,));
        constructors.push(quote! {
            #[doc(hidden)]
            pub fn #new() -> #hw_name {
                let mut value = <#name as #dsl::HwType>::zero();
                value.__tag = #dsl::Val::constant(#tag);
                value
            }
            #[doc(hidden)]
            pub fn #is(&self) -> #dsl::Val<bool> {
                #dsl::eq(self.__tag, #dsl::Val::constant(#tag))
            }
            #[doc(hidden)]
            pub fn #same(&self, other: &#hw_name) -> #dsl::Val<bool> {
                let (mut left, mut right) = (::std::vec::Vec::new(), ::std::vec::Vec::new());
                #(
                    <#field_types as #dsl::HwType>::outputs(&self.#fields, &mut left);
                    <#field_types as #dsl::HwType>::outputs(&other.#fields, &mut right);
                )*
                #dsl::cells_equal(&left, &right)
            }
        });
        conversions.push(quote! {
            #bind => {
                let mut value = #hw_name::#new();
                #(value.#fields = #dsl::Conv::conv(#bindings);)*
                value
            }
        });
        native_equal.push(if fields.is_empty() {
            quote!(#name::#ident { .. } => self.#is(),)
        } else {
            quote!(#name::#ident { .. } => #dsl::and(self.#is(), || self.#same(&other)),)
        });
        if !fields.is_empty() {
            circuit_equal.push(quote! {
                bit = #dsl::and(bit, || #dsl::or(!self.#is(), || self.#same(&right)));
            });
        }
    }
    let doc = format!(
        "The circuit form of [`{name}`], for `#[handwired]` code: which variant it is, and each variant's fields."
    );

    Ok(quote! {
        #[doc = #doc]
        #[derive(Copy, Clone)]
        #[allow(dead_code, non_snake_case)]
        #visibility struct #hw_name {
            #[doc(hidden)]
            pub __tag: #dsl::Val<u8>,
            #(#[doc(hidden)] pub #hw_fields: <#types as #dsl::HwType>::Val,)*
        }

        #[allow(non_snake_case, clippy::all)]
        impl #hw_name {
            #(#constructors)*

            /// The smaller of the two, in the order Rust derives: `self` where they're equal, like `Ord::min`.
            pub fn min(self, other: impl #dsl::Conv<#hw_name>) -> #hw_name {
                let other: #hw_name = #dsl::Conv::conv(other);
                #dsl::Mux::select(#dsl::le(self, other), other, self)
            }
            /// The larger of the two, in the order Rust derives: `other` where they're equal, like `Ord::max`.
            pub fn max(self, other: impl #dsl::Conv<#hw_name>) -> #hw_name {
                let other: #hw_name = #dsl::Conv::conv(other);
                #dsl::Mux::select(#dsl::gt(self, other), other, self)
            }
            /// `self` within `low..=high`, like `Ord::clamp`.
            pub fn clamp(self, low: impl #dsl::Conv<#hw_name>, high: impl #dsl::Conv<#hw_name>) -> #hw_name {
                self.max(low).min(high)
            }
        }

        #[allow(unused_assignments, unused_variables, unused_mut, unreachable_patterns, clippy::all)]
        const _: () = {
            const FIELD_ATTRS: [#dsl::Attrs; #count] = [#(#attrs),*];
            const TAGS: [u8; #variant_count] = [#(#tags),*];

            /// The tag's attributes: its entry of the parameter's `ranges` when given, else every variant.
            fn tag_attrs(attrs: &#dsl::Attrs) -> #dsl::Attrs {
                if attrs.ranges.is_some() {
                    attrs.part(0, 1)
                } else {
                    #dsl::Attrs::new(Some((0.0, #last)), None, None)
                }
            }

            impl #dsl::HwType for #name {
                type Val = #hw_name;
                const CELLS: usize = 1 #(+ <#types as #dsl::HwType>::CELLS)*;
                fn declare(inputs: &mut #dsl::Inputs, attrs: &#dsl::Attrs) -> #hw_name {
                    attrs.expect_cells(<Self as #dsl::HwType>::CELLS);
                    let tag = tag_attrs(attrs);
                    #hw_name {
                        __tag: #dsl::declare_scalar::<u8>(inputs, &if attrs.holds_zero { tag.holding_zero() } else { tag }),
                        #(
                            #hw_fields: {
                                let cells = <#types as #dsl::HwType>::CELLS;
                                let field = attrs.field(&FIELD_ATTRS[#field_indices], #offsets, cells).holding_zero();
                                <#types as #dsl::HwType>::declare(inputs, &field)
                            },
                        )*
                    }
                }
                fn encode(&self, out: &mut Vec<f32>) {
                    let tag: u8 = match self { #(#binds => #tags,)* };
                    out.push(tag as f32);
                    #(#encode_fields)*
                }
                fn decode(values: &mut dyn Iterator<Item = f32>) -> Self {
                    let tag = values.next().expect("too few output values");
                    let mut result = None;
                    #(#decode_variants)*
                    result.unwrap_or_else(|| panic!("{} has no variant {}", #label, tag))
                }
                fn outputs(value: &#hw_name, out: &mut Vec<::handwired_rs::Wire>) {
                    out.push(value.__tag.wire());
                    #(<#types as #dsl::HwType>::outputs(&value.#hw_fields, out);)*
                }
                fn sample(rng: &mut #dsl::Rng, attrs: &#dsl::Attrs) -> Self {
                    attrs.expect_cells(<Self as #dsl::HwType>::CELLS);
                    match #dsl::sample_variant(rng, &tag_attrs(attrs), &TAGS) {
                        #(#sample_variants)*
                        _ => unreachable!(),
                    }
                }
                fn zero() -> #hw_name {
                    #hw_name {
                        __tag: #dsl::Val::constant(0u8),
                        #(#hw_fields: <#types as #dsl::HwType>::zero(),)*
                    }
                }
            }

            impl #dsl::Native for #hw_name {
                type Of = #name;
            }
            impl #dsl::Mux for #hw_name {
                fn select(bit: #dsl::Val<bool>, off: Self, on: Self) -> Self {
                    #hw_name {
                        __tag: #dsl::Mux::select(bit, off.__tag, on.__tag),
                        #(#hw_fields: #dsl::Mux::select(bit, off.#hw_fields, on.#hw_fields),)*
                    }
                }
            }
            impl #dsl::Placeholder for #hw_name {
                fn placeholder() -> Self {
                    <#name as #dsl::HwType>::zero()
                }
            }
            impl #dsl::Conv<#hw_name> for #hw_name {
                fn conv(self) -> #hw_name {
                    self
                }
            }
            impl #dsl::Conv<#hw_name> for #name {
                fn conv(self) -> #hw_name {
                    match self { #(#conversions)* }
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

            impl #dsl::Cmp<#hw_name> for #hw_name {
                type Output = #dsl::Val<bool>;
                fn compare(self, relation: ::handwired_rs::Relation, right: #hw_name) -> #dsl::Val<bool> {
                    // The variant in declaration order, then its fields; other variants' cells are 0 on both sides.
                    let (mut left_cells, mut right_cells) = (::std::vec::Vec::new(), ::std::vec::Vec::new());
                    <#name as #dsl::HwType>::outputs(&self, &mut left_cells);
                    <#name as #dsl::HwType>::outputs(&right, &mut right_cells);
                    #dsl::cells_compare(&left_cells, relation, &right_cells)
                }
                fn equal(self, right: #hw_name, equal: bool) -> #dsl::Val<bool> {
                    // The same variant, and where it's a variant with fields, the same fields.
                    let mut bit = #dsl::eq(self.__tag, right.__tag);
                    #(#circuit_equal)*
                    if equal { bit } else { !bit }
                }
            }
            impl #dsl::Cmp<#name> for #hw_name {
                type Output = #dsl::Val<bool>;
                fn compare(self, relation: ::handwired_rs::Relation, right: #name) -> #dsl::Val<bool> {
                    #dsl::Cmp::compare(self, relation, <#name as #dsl::Conv<#hw_name>>::conv(right))
                }
                fn equal(self, right: #name, equal: bool) -> #dsl::Val<bool> {
                    // The right side's variant is known while the circuit is built.
                    let other: #hw_name = #dsl::Conv::conv(right);
                    let bit = match right { #(#native_equal)* };
                    if equal { bit } else { !bit }
                }
            }
            impl #dsl::Cmp<#hw_name> for #name {
                type Output = #dsl::Val<bool>;
                fn compare(self, relation: ::handwired_rs::Relation, right: #hw_name) -> #dsl::Val<bool> {
                    #dsl::Cmp::compare(<#name as #dsl::Conv<#hw_name>>::conv(self), relation, right)
                }
                fn equal(self, right: #hw_name, equal: bool) -> #dsl::Val<bool> {
                    #dsl::Cmp::equal(right, self, equal)
                }
            }
            impl #dsl::Cmp<#name> for #name {
                type Output = bool;
                fn compare(self, relation: ::handwired_rs::Relation, right: #name) -> bool {
                    // By the encoded cells, as the circuit compares, so it needs no `PartialOrd`.
                    #dsl::compare_encoded(&self, relation, &right)
                }
                fn equal(self, right: #name, equal: bool) -> bool {
                    (self == right) == equal
                }
            }
        };
    })
}
