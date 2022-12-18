use std::{ops::Range, path::Path, rc::Rc};

use chumsky::{prelude::*, text::whitespace};

use super::ast::{
    CallName, Definition, Disjunction, Goal, GoalList, Name, Program, Query, SourceId, Term,
    TermList,
};

pub type ParseError = Simple<char, (SourceId, Range<usize>)>;

fn lowercase(c: &char) -> bool {
    c.is_lowercase()
}

fn uppercase(c: &char) -> bool {
    c.is_uppercase()
}

fn ident(c: &char) -> bool {
    c.is_alphanumeric() || *c == '_'
}

macro_rules! operators {
    ($operator:literal) => {
        just($operator)
    };

    ($($operator:literal),*) => {
        choice(($(just($operator)),*))
    };
}

fn maybe_prefix(
    term: impl Parser<char, Term, Error = ParseError>,
    operators: impl Parser<char, &'static str, Error = ParseError>,
) -> impl Parser<char, Term, Error = ParseError> {
    operators
        .map_with_span(Name::new)
        .then_ignore(text::whitespace())
        .or_not()
        .then(term)
        .map(|(operator, term)| match operator {
            Some(operator) => Term::Structure {
                name: operator,
                terms: TermList::from([term]),
            },
            None => term,
        })
}

fn infix_many(
    term: impl Parser<char, Term, Error = ParseError> + 'static,
    operators: impl Parser<char, &'static str, Error = ParseError>,
) -> impl Parser<char, Term, Error = ParseError> {
    let term = term.boxed();

    term.clone()
        .then(
            operators
                .map_with_span(Name::new)
                .padded()
                .then(term)
                .repeated(),
        )
        .foldl(|lhs, (name, rhs)| Term::Structure {
            name,
            terms: TermList::from([lhs, rhs]),
        })
}

fn name() -> impl Parser<char, Name, Error = ParseError> {
    filter(lowercase)
        .chain(filter(ident).repeated())
        .collect::<String>()
        .map_with_span(Name::new)
        .labelled("name")
}

fn make_structure(
    term: impl Parser<char, Term, Error = ParseError>,
) -> impl Parser<char, (Name, TermList), Error = ParseError> {
    name()
        .then_ignore(whitespace())
        .then(
            term.separated_by(just(',').padded())
                .padded()
                .delimited_by(just('('), just(')'))
                .or_not()
                .map(Option::unwrap_or_default)
                .map(TermList::from),
        )
        .labelled("structure")
}

fn term() -> impl Parser<char, Term, Error = ParseError> {
    recursive(|term| {
        let variable = filter(uppercase)
            .chain(filter(ident).repeated())
            .collect::<String>()
            .map_with_span(Name::new)
            .map(|name| Term::Variable { name })
            .labelled("variable");

        let structure = make_structure(term.clone()).map(|(name, terms)| {
            if terms.is_empty() {
                Term::Constant { name }
            } else {
                Term::Structure { name, terms }
            }
        });

        let list = term
            .clone()
            .separated_by(just(',').padded())
            .then(
                just('|')
                    .padded()
                    .ignore_then(term.clone())
                    .or(empty().map_with_span(|(), span| Term::Constant {
                        name: Name::new("[]", span),
                    })),
            )
            .padded()
            .delimited_by(just('['), just(']'))
            .foldr(Term::list);

        let integer = just('-')
            .or_not()
            .chain::<char, _, _>(text::int(10))
            .collect::<String>()
            .from_str()
            .unwrapped()
            .map(|i| Term::Integer { i })
            .labelled("integer");

        let void = just('_')
            .then(filter(ident).repeated())
            .ignored()
            .map(|()| Term::Void)
            .labelled("void");

        let term = variable
            .or(structure)
            .or(list)
            .or(integer)
            .or(void)
            .or(term.padded().delimited_by(just('('), just(')')));

        let term = maybe_prefix(term, operators!("+", "-"));
        let term = infix_many(term, operators!("*", "//", "div", "mod"));
        let term = infix_many(term, operators!("+", "-"));
        let term = infix_many(term, operators!(":"));

        #[allow(clippy::let_and_return)]
        term
    })
}

trait FromCallAndTerms: Sized {
    fn from_call_and_terms(name: CallName<Name>, terms: TermList) -> Self;
}

impl FromCallAndTerms for Goal<Name> {
    fn from_call_and_terms(name: CallName<Name>, terms: TermList) -> Self {
        Goal::Named { name, terms }
    }
}

impl FromCallAndTerms for Query {
    fn from_call_and_terms(name: CallName<Name>, terms: TermList) -> Self {
        Query { name, terms }
    }
}

fn true_or_fail_or_named_call<T: FromCallAndTerms>() -> impl Parser<char, T, Error = ParseError> {
    make_structure(term()).map(|(name, terms)| {
        let name = match (name.as_ref(), terms.len()) {
            ("true", 0) => CallName::True,
            ("false" | "fail", 0) => CallName::Fail,
            _ => CallName::Named(name),
        };

        T::from_call_and_terms(name, terms)
    })
}

fn is() -> impl Parser<char, TermList, Error = ParseError> {
    let term = term().boxed();

    term.clone()
        .then_ignore(just("is").padded())
        .then(term)
        .map(|(lhs, rhs)| TermList::from([lhs, rhs]))
}

pub fn parse_program(path: Rc<Path>, program_source: &str) -> Result<Program, Vec<ParseError>> {
    let source_id = SourceId::Program(path);

    let goal = true_or_fail_or_named_call()
        .or(just('!').map(|_| Goal::Cut))
        .or(is().map(|terms| Goal::Named {
            name: CallName::Is,
            terms,
        }));

    let rule = make_structure(term())
        .then_ignore(just(":-").padded())
        .then(
            goal.padded()
                .separated_by(just(','))
                .at_least(1)
                .map(GoalList::from),
        )
        .then_ignore(just('.'))
        .map(|((name, head), body)| (name, Definition { head, body }));

    let fact = make_structure(term())
        .then_ignore(just('.'))
        .map(|(name, head)| {
            (
                name,
                Definition {
                    head,
                    body: GoalList::new(),
                },
            )
        });

    let definition = rule.or(fact).padded();

    definition
        .repeated()
        .then_ignore(end())
        .map(|definitions| {
            let mut definitions = definitions
                .into_iter()
                .map(|(name, definition)| {
                    (
                        name,
                        definition.with_singletons_removed(source_id.clone(), program_source),
                    )
                })
                .peekable();

            Program {
                definitions: core::iter::from_fn(|| {
                    let (first_name, first_definition) = definitions.next()?;

                    let mut middle_definitions = Vec::new();

                    while let Some((_, definition)) = definitions.next_if(|(name, definition)| {
                        (name, definition.head.len()) == (&first_name, first_definition.head.len())
                    }) {
                        middle_definitions.push(definition);
                    }

                    Some(match middle_definitions.pop() {
                        None => Disjunction::Single {
                            name: first_name,
                            arity: first_definition.head.len() as u8,
                            definition: first_definition,
                        },
                        Some(last_definition) => Disjunction::Multiple {
                            name: first_name,
                            arity: first_definition.head.len() as u8,
                            first_definition,
                            middle_definitions,
                            last_definition,
                        },
                    })
                })
                .collect(),
            }
        })
        .parse(source_id.clone().stream(program_source))
}

pub fn parse_query(query: &str) -> Result<Query, Vec<ParseError>> {
    true_or_fail_or_named_call()
        .or(is().map(|terms| Query {
            name: CallName::Is,
            terms,
        }))
        .padded()
        .then_ignore(just('.').padded())
        .then_ignore(end())
        .parse(SourceId::Query.stream(query))
}

pub fn print_parse_error_report(error: ParseError, program_info: &super::ProgramInfo) {
    use ariadne::{Color, Fmt, Label, ReportKind};

    let msg = if let chumsky::error::SimpleReason::Custom(msg) = error.reason() {
        msg.clone()
    } else {
        format!(
            "{}{}, expected {}",
            if error.found().is_some() {
                "Unexpected token"
            } else {
                "Unexpected end of input"
            },
            if let Some(label) = error.label() {
                format!(" while parsing {}", label)
            } else {
                String::new()
            },
            if error.expected().len() == 0 {
                "something else".to_string()
            } else {
                error
                    .expected()
                    .map(|expected| match expected {
                        Some(expected) => expected.to_string(),
                        None => "end of input".to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            },
        )
    };

    let report = ariadne::Report::build(
        ReportKind::Error,
        error.span().context(),
        error.span().start(),
    )
    .with_message(msg)
    .with_label(
        Label::new(error.span())
            .with_message(match error.reason() {
                chumsky::error::SimpleReason::Custom(msg) => msg.clone(),
                _ => format!(
                    "Unexpected {}",
                    error
                        .found()
                        .map(|c| format!("token {}", c.fg(Color::Red)))
                        .unwrap_or_else(|| "end of input".to_string())
                ),
            })
            .with_color(Color::Red),
    );

    let report = match error.reason() {
        chumsky::error::SimpleReason::Unclosed { span, delimiter } => report.with_label(
            Label::new(span.clone())
                .with_message(format!(
                    "Unclosed delimiter {}",
                    delimiter.fg(Color::Yellow)
                ))
                .with_color(Color::Yellow),
        ),
        chumsky::error::SimpleReason::Unexpected => report,
        chumsky::error::SimpleReason::Custom(_) => report,
    };

    report
        .finish()
        .eprint(program_info.as_source_cache())
        .unwrap();
}
