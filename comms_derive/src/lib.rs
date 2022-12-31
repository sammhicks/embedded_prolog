extern crate proc_macro;

use itertools::Itertools;
use proc_macro::TokenStream;
use syn::{parse_macro_input, DeriveInput};

fn parse_newtype_field(data: syn::Data) -> syn::Field {
    let syn::Data::Struct(syn::DataStruct { fields, .. }) = data else {
        panic!("Only structs are supported")
    };

    match fields {
        syn::Fields::Unnamed(syn::FieldsUnnamed { unnamed, .. }) => unnamed
            .into_iter()
            .exactly_one()
            .unwrap_or_else(|_| panic!("Only structs with a single field are supported")),
        syn::Fields::Named(syn::FieldsNamed { named, .. }) => named
            .into_iter()
            .exactly_one()
            .unwrap_or_else(|_| panic!("Only structs with a single field are supported")),
        syn::Fields::Unit => panic!("Only structs with a single field are supported"),
    }
}

fn parse_newtype_format_info(data: syn::Data) -> (proc_macro2::TokenStream, String, &'static str) {
    let field = parse_newtype_field(data);

    let field_name = field
        .ident
        .map_or_else(|| quote::quote!(self.0), |name| quote::quote!(self.#name));

    let field_type = match field.ty {
        syn::Type::Path(syn::TypePath {
            path: syn::Path { segments, .. },
            ..
        }) => segments
            .into_iter()
            .exactly_one()
            .unwrap_or_else(|_| panic!("Must be plain identifier"))
            .ident
            .to_string(),
        _ => panic!("Must be plain identifier"),
    };

    let field_format = match field_type.as_str() {
        "u8" => r"{:02X}",
        "u16" | "NonZeroU16" => r"{:04X}",
        _ => panic!("Type {} not handled", field_type),
    };

    (field_name, field_type, field_format)
}

#[proc_macro_derive(HexNewType)]
pub fn derive_hex_newtype(input: TokenStream) -> TokenStream {
    let DeriveInput {
        ident: struct_name_ident,
        data,
        ..
    } = parse_macro_input!(input as DeriveInput);

    let (field_name, _, field_format) = parse_newtype_format_info(data);

    let debug_format = format!("{}({{}})", struct_name_ident);

    quote::quote!(
        impl core::fmt::Debug for #struct_name_ident {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, #debug_format, self)
            }
        }

        impl core::fmt::Display for #struct_name_ident {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, #field_format, #field_name)
            }
        }
    )
    .into()
}

#[proc_macro_derive(HexDefmt)]
pub fn derive_hex_defmt(input: TokenStream) -> TokenStream {
    let DeriveInput {
        ident: struct_name_ident,
        data,
        ..
    } = parse_macro_input!(input as DeriveInput);

    let (field_name, field_type, field_format) = parse_newtype_format_info(data);

    let field_name = if field_type.starts_with("NonZero") {
        quote::quote!(#field_name.get())
    } else {
        field_name
    };

    quote::quote!(
        impl defmt::Format for #struct_name_ident {
            fn format(&self, fmt: defmt::Formatter) {
                defmt::write!(fmt, #field_format, #field_name)
            }
        }
    )
    .into()
}

#[proc_macro_derive(CommsFromInto)]
pub fn derive_comms_from_into(input: TokenStream) -> TokenStream {
    let DeriveInput {
        ident: struct_name_ident,
        data,
        ..
    } = parse_macro_input!(input as DeriveInput);

    let field = parse_newtype_field(data);

    match field.ident {
        None => {
            quote::quote!(
                impl From<comms::#struct_name_ident> for #struct_name_ident {
                    fn from(::comms::#struct_name_ident(inner): ::comms::#struct_name_ident) -> Self {
                        Self(inner)
                    }
                }
        
                impl From<#struct_name_ident> for ::comms::#struct_name_ident {
                    fn from(#struct_name_ident(inner): #struct_name_ident) -> Self {
                        Self(inner)
                    }
                }
            )
            .into()
        }
        Some(field_name) => {
            quote::quote!(
                impl From<comms::#struct_name_ident> for #struct_name_ident {
                    fn from(::comms::#struct_name_ident{ #field_name }: ::comms::#struct_name_ident) -> Self {
                        Self{ #field_name }
                    }
                }
        
                impl From<#struct_name_ident> for ::comms::#struct_name_ident {
                    fn from(#struct_name_ident{ #field_name }: #struct_name_ident) -> Self {
                        Self{ #field_name }
                    }
                }
            )
            .into()
        }
    }
}
