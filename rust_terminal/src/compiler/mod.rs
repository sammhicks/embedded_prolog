use std::{
    borrow::Borrow,
    collections::{BTreeMap, HashMap, HashSet},
    fmt,
    hash::Hash,
    path::{Path, PathBuf},
    rc::Rc,
};

mod ast;
mod instructions;
mod parser;

use arcstr::ArcStr;
use ast::{CallName, Definition, Disjunction, Goal, SourceId, Term, TermList, VariableName};
use instructions::{Ai, Instruction, LongInteger, ShortInteger, Xn, Yn};

pub use ast::{Name, Query, EMPTY_LIST};
pub use instructions::{Arity, Functor};
use num_bigint::BigInt;

trait IntoOkOrError {
    type Output;

    fn into_ok_or_error(self) -> Self::Output;
}

impl<T> IntoOkOrError for Result<T, T> {
    type Output = T;

    fn into_ok_or_error(self) -> Self::Output {
        match self {
            Ok(t) | Err(t) => t,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FunctorSet(BTreeMap<ArcStr, u16>);

impl FunctorSet {
    pub fn get(&mut self, functor: ArcStr) -> u16 {
        let len = self.0.len() as u16;

        *self.0.entry(functor).or_insert(len)
    }

    pub fn lookup(&self, functor: &u16) -> Option<&str> {
        self.0
            .iter()
            .find_map(|(name, entry)| (entry == functor).then_some(name.as_str()))
    }
}

impl<S> Extend<S> for FunctorSet
where
    ArcStr: From<S>,
{
    fn extend<T: IntoIterator<Item = S>>(&mut self, functors: T) {
        for functor in functors {
            self.get(functor.into());
        }
    }
}

impl<const N: usize> From<[ArcStr; N]> for FunctorSet {
    fn from(functors: [ArcStr; N]) -> Self {
        Self(IntoIterator::into_iter(functors).zip(0..).collect())
    }
}

struct ProgramInfoSourceCache<'a> {
    program_info: &'a ProgramInfo,
}

impl<'a> ariadne::Cache<SourceId> for ProgramInfoSourceCache<'a> {
    fn fetch(&mut self, id: &SourceId) -> Result<&ariadne::Source, Box<dyn fmt::Debug + '_>> {
        self.program_info
            .source_files
            .get(id)
            .map(|source| source.as_ref())
            .ok_or_else(|| Box::new(format!("No source for {id}")) as Box<dyn fmt::Debug>)
    }

    fn display<'b>(&self, id: &'b SourceId) -> Option<Box<dyn fmt::Display + 'b>> {
        Some(Box::new(id))
    }
}

#[derive(Clone)]
pub struct ProgramInfo {
    source_files: HashMap<SourceId, Rc<ariadne::Source>>,
    labels: HashMap<(Name, Arity), u16>,
    functors: FunctorSet,
}

impl ProgramInfo {
    pub fn new(system_call_source: &str, program_path: Rc<Path>, program_source: &str) -> Self {
        Self {
            source_files: [
                (
                    SourceId::SystemCall,
                    Rc::new(ariadne::Source::from(system_call_source)),
                ),
                (
                    SourceId::Program(program_path),
                    Rc::new(ariadne::Source::from(program_source)),
                ),
            ]
            .into(),
            labels: HashMap::new(),
            functors: [
                arcstr::literal!("+"),
                arcstr::literal!("-"),
                arcstr::literal!("*"),
                arcstr::literal!("//"),
                arcstr::literal!("div"),
                arcstr::literal!("mod"),
                arcstr::literal!("min"),
                arcstr::literal!("max"),
                arcstr::literal!("clamp"),
            ]
            .into(),
        }
    }

    fn with_query_source(&self, query_source: &str) -> Self {
        let mut new_info = self.clone();
        new_info.source_files.insert(
            SourceId::Query,
            Rc::new(ariadne::Source::from(query_source)),
        );
        new_info
    }

    pub fn lookup_functor(&self, name: &u16) -> Option<&str> {
        self.functors.lookup(name)
    }

    fn as_source_cache(&self) -> ProgramInfoSourceCache {
        ProgramInfoSourceCache { program_info: self }
    }
}

trait SortExt<T>: Sized + std::ops::DerefMut<Target = [T]> {
    fn sorted_by_key<K: Ord>(mut self, f: impl FnMut(&T) -> K) -> Self {
        self.sort_by_key(f);
        self
    }
}

impl<T, C: std::ops::DerefMut<Target = [T]>> SortExt<T> for C {}

struct DisplayList<I>(I);

impl<I> fmt::Display for DisplayList<I>
where
    I: Copy + IntoIterator,
    I::Item: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;

        let mut items = self.0.into_iter();

        if let Some(item) = items.next() {
            write!(f, "{}", item)?;
        }

        for item in items {
            write!(f, ", {item}")?;
        }

        write!(f, "]")
    }
}

trait Register {
    fn new(index: u8) -> Self;
}

impl Register for Xn {
    fn new(xn: u8) -> Self {
        Self { xn }
    }
}

impl Register for Ai {
    fn new(ai: u8) -> Self {
        Self { ai }
    }
}

impl Register for Vn {
    fn new(xn: u8) -> Self {
        Vn::Xn { xn: Xn { xn } }
    }
}

#[derive(Debug)]

struct NextFreeRegister(u8);

impl NextFreeRegister {
    fn new() -> Self {
        Self(0)
    }

    fn next<R: Register>(&mut self) -> R {
        let index = self.0;
        self.0 += 1;
        R::new(index)
    }
}

trait CompilationMode: Sized {
    type Variable: fmt::Debug + fmt::Display + Copy + Eq + Hash + Register;
}

trait IsQueryMode: CompilationMode {}

struct ProgramMode;

impl CompilationMode for ProgramMode {
    type Variable = Vn;
}

struct RuleGoalMode;

impl CompilationMode for RuleGoalMode {
    type Variable = Vn;
}

impl IsQueryMode for RuleGoalMode {}

struct QueryMode;

impl CompilationMode for QueryMode {
    type Variable = Xn;
}

impl IsQueryMode for QueryMode {}

#[derive(Debug)]
enum TermAllocation<'a, Mode: CompilationMode> {
    Variable {
        vn: Mode::Variable,
        name: &'a Name,
    },
    Structure {
        xn: Xn,
        name: &'a Name,
        terms: Vec<TermAllocation<'a, Mode>>,
    },
    List {
        xn: Xn,
        terms: Vec<TermAllocation<'a, Mode>>,
    },
    Constant {
        name: &'a Name,
    },
    Integer {
        i: &'a BigInt,
    },
    Void,
}

impl<'a, Mode: CompilationMode> fmt::Display for TermAllocation<'a, Mode> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TermAllocation::Variable { vn, name } => write!(f, "{vn}={name}"),
            TermAllocation::Structure { xn, name, terms } => {
                write!(
                    f,
                    "{}=s({}/{}, {})",
                    xn,
                    name,
                    terms.len(),
                    DisplayList(terms)
                )
            }
            TermAllocation::List { xn, terms } => write!(f, "{}=l{}", xn, DisplayList(terms)),
            TermAllocation::Constant { name } => write!(f, "c({name})"),
            TermAllocation::Integer { i } => write!(f, "i({i})"),
            TermAllocation::Void => write!(f, "_"),
        }
    }
}

#[derive(Debug)]
enum ArgumentAllocation<'a, Mode: CompilationMode> {
    Variable {
        ai: Ai,
        vn: Mode::Variable,
        name: &'a Name,
    },
    Structure {
        ai: Ai,
        name: &'a Name,
        terms: Vec<TermAllocation<'a, Mode>>,
    },
    List {
        ai: Ai,
        terms: Vec<TermAllocation<'a, Mode>>,
    },
    Constant {
        ai: Ai,
        name: &'a Name,
    },
    Integer {
        ai: Ai,
        i: &'a BigInt,
    },
    Void {
        ai: Ai,
    },
}

impl<'a, Mode: CompilationMode> fmt::Display for ArgumentAllocation<'a, Mode> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArgumentAllocation::Variable { ai, vn, name } => {
                write!(f, "{ai}=({vn}={name})")
            }
            ArgumentAllocation::Structure { ai, name, terms } => write!(
                f,
                "{}=s({}/{}, {})",
                ai,
                name,
                terms.len(),
                DisplayList(terms)
            ),
            ArgumentAllocation::List { ai, terms } => write!(f, "{}=l{}", ai, DisplayList(terms)),
            ArgumentAllocation::Constant { ai, name } => write!(f, "{ai}={name}"),
            ArgumentAllocation::Integer { ai, i } => write!(f, "{ai}={i}"),
            ArgumentAllocation::Void { ai } => write!(f, "{ai}=_"),
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
enum Vn {
    Xn { xn: Xn },
    Yn { yn: Yn },
}

impl fmt::Display for Vn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Vn::Xn { xn } => write!(f, "{xn}"),
            Vn::Yn { yn } => write!(f, "{yn}"),
        }
    }
}

impl From<Ai> for Vn {
    fn from(ai: Ai) -> Self {
        Vn::from(Xn::from(ai))
    }
}

impl From<Xn> for Vn {
    fn from(xn: Xn) -> Self {
        Self::Xn { xn }
    }
}

impl From<Yn> for Vn {
    fn from(yn: Yn) -> Self {
        Self::Yn { yn }
    }
}

struct PermanentVariablesAllocation<'a> {
    permanent_variables: Vec<VariableName<'a>>,
    already_declared_permanent_variables: Vec<HashSet<VariableName<'a>>>,
    trimmed_permanent_variables: Vec<u8>,
}

impl<'a> PermanentVariablesAllocation<'a> {
    fn new(head: &'a TermList, goals: &'a [Goal<Name>]) -> Self {
        let head_variables = head.variables(HashSet::from([VariableName::Cut]));
        let goals_variables = goals
            .iter()
            .map(|goal| goal.variables(HashSet::new()))
            .collect::<Vec<_>>();

        let head_permanent_variables = head_variables
            .intersection(
                &goals_variables
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<HashSet<_>>(),
            )
            .copied()
            .collect::<HashSet<_>>();

        let goals_permanent_variables = {
            let mut goal_variables = goals_variables.iter();

            std::iter::from_fn(|| {
                let variable_set = goal_variables.next()?;

                let following_variables = goal_variables
                    .as_slice()
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<HashSet<_>>();

                Some(
                    variable_set
                        .intersection(&following_variables)
                        .copied()
                        .collect(),
                )
            })
            .collect::<Vec<_>>()
        };

        let all_permanent_variables = std::iter::once(&head_permanent_variables)
            .chain(goals_permanent_variables.iter())
            .flatten()
            .collect::<HashSet<_>>();

        let permanent_variables_by_last_goal = (0_i32..)
            .zip(goals_variables.iter())
            .fold(
                HashMap::new(),
                |mut variables_sorted_by_last_mention, (index, variables)| {
                    for variable in variables.iter().copied() {
                        if all_permanent_variables.contains(&variable) {
                            variables_sorted_by_last_mention.insert(variable, index);
                        }
                    }

                    variables_sorted_by_last_mention
                },
            )
            .into_iter()
            .collect::<Vec<_>>()
            .sorted_by_key(|&(_, index)| -index);

        let permanent_variables = permanent_variables_by_last_goal
            .iter()
            .map(|(variable, _)| variable)
            .copied()
            .collect();

        let already_declared_permanent_variables = goals_permanent_variables
            .iter()
            .scan(
                head_permanent_variables.clone(),
                |already_declared_permanent_variables, newly_declared_permanent_variables| {
                    let mut now_declared_permanent_variables = already_declared_permanent_variables
                        .union(newly_declared_permanent_variables)
                        .copied()
                        .collect();

                    std::mem::swap(
                        already_declared_permanent_variables,
                        &mut now_declared_permanent_variables,
                    );

                    Some(now_declared_permanent_variables)
                },
            )
            .collect();

        let mut trimmed_permanent_variables = (0_i32..)
            .take(goals.len())
            .map(|index| {
                permanent_variables_by_last_goal
                    .iter()
                    .filter_map(|(_, permanent_variable_index)| {
                        (index == *permanent_variable_index).then_some(1)
                    })
                    .sum()
            })
            .collect::<Vec<_>>();

        // There's no point the last goal trimming the stack as its about to be deallocated
        if let Some(last) = trimmed_permanent_variables.last_mut() {
            *last = 0;
        }

        Self {
            permanent_variables,
            already_declared_permanent_variables,
            trimmed_permanent_variables,
        }
    }
}

struct RegisterAllocationState<'a, Mode: CompilationMode> {
    next_free_register: NextFreeRegister,
    variables: HashMap<VariableName<'a>, Mode::Variable>,
}

impl<'a, Mode: CompilationMode> RegisterAllocationState<'a, Mode> {
    fn new() -> Self {
        Self {
            next_free_register: NextFreeRegister::new(),
            variables: HashMap::new(),
        }
    }
}

impl<'a> RegisterAllocationState<'a, ProgramMode> {
    fn program() -> Self {
        Self::new()
    }
}

impl<'a> RegisterAllocationState<'a, QueryMode> {
    fn query() -> Self {
        Self::new()
    }
}

impl<'a, Mode: CompilationMode<Variable = Vn>> RegisterAllocationState<'a, Mode> {
    fn with_permanent_variables(permenent_variables: &[VariableName<'a>]) -> Self {
        Self {
            next_free_register: NextFreeRegister::new(),
            variables: permenent_variables
                .iter()
                .copied()
                .zip((0..).map(|yn| Vn::Yn { yn: Yn { yn } }))
                .collect(),
        }
    }
}

enum TermRegisterAllocation<'a, Mode: CompilationMode> {
    Variable {
        vn: Mode::Variable,
        name: &'a Name,
    },
    Structure {
        xn: Xn,
        name: &'a Name,
        terms: &'a [Term],
    },
    Constant {
        name: &'a Name,
    },
    List {
        xn: Xn,
        head: &'a Term,
        tail: &'a Term,
    },
    Integer {
        i: &'a BigInt,
    },
    Void,
}

impl<'a, Mode: CompilationMode> RegisterAllocationState<'a, Mode> {
    fn reserve_variable_register(&mut self, name: &'a Name) -> Mode::Variable {
        *self
            .variables
            .entry(VariableName::Named(name))
            .or_insert_with(|| self.next_free_register.next())
    }

    fn allocate_structure_terms_registers<T: IntoIterator<Item = &'a Term>>(
        &mut self,
        terms: T,
    ) -> Vec<TermAllocation<'a, Mode>> {
        let term_allocation = terms
            .into_iter()
            .map(|term| match term.borrow() {
                Term::Variable { name } => TermRegisterAllocation::Variable {
                    vn: self.reserve_variable_register(name),
                    name,
                },
                Term::Structure { name, terms } => TermRegisterAllocation::Structure {
                    xn: self.next_free_register.next(),
                    name,
                    terms,
                },
                Term::List { head, tail } => TermRegisterAllocation::List {
                    xn: self.next_free_register.next(),
                    head,
                    tail,
                },
                Term::Constant { name } => TermRegisterAllocation::Constant { name },
                Term::Integer { i } => TermRegisterAllocation::Integer { i },
                Term::Void => TermRegisterAllocation::Void,
            })
            .collect::<Vec<TermRegisterAllocation<Mode>>>();

        term_allocation
            .iter()
            .map(|allocation| match *allocation {
                TermRegisterAllocation::Variable { vn, name } => {
                    TermAllocation::Variable { vn, name }
                }
                TermRegisterAllocation::Structure { xn, name, terms } => {
                    TermAllocation::Structure {
                        xn,
                        name,
                        terms: self.allocate_structure_terms_registers(terms),
                    }
                }
                TermRegisterAllocation::List { xn, head, tail } => TermAllocation::List {
                    xn,
                    terms: self.allocate_structure_terms_registers([head, tail]),
                },
                TermRegisterAllocation::Constant { name } => TermAllocation::Constant { name },
                TermRegisterAllocation::Integer { i } => TermAllocation::Integer { i },
                TermRegisterAllocation::Void => TermAllocation::Void,
            })
            .collect()
    }

    fn allocate_argument_registers(mut self, terms: &'a [Term]) -> Vec<ArgumentAllocation<Mode>> {
        let arguments_allocation = terms
            .iter()
            .map(|term| (self.next_free_register.next(), term))
            .collect::<Vec<_>>();

        arguments_allocation
            .into_iter()
            .map(|(ai, term)| match term {
                Term::Variable { name } => ArgumentAllocation::Variable {
                    ai,
                    vn: self.reserve_variable_register(name),
                    name,
                },
                Term::Structure { name, terms } => ArgumentAllocation::Structure {
                    ai,
                    name,
                    terms: self.allocate_structure_terms_registers(terms),
                },
                Term::List { head, tail } => ArgumentAllocation::List {
                    ai,
                    terms: self.allocate_structure_terms_registers([head.as_ref(), tail.as_ref()]),
                },
                Term::Constant { name } => ArgumentAllocation::Constant { ai, name },
                Term::Integer { i } => ArgumentAllocation::Integer { ai, i },
                Term::Void => ArgumentAllocation::Void { ai },
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Label<'a> {
    name: &'a Name,
    arity: Arity,
    retry: usize,
}

impl<'a> Label<'a> {
    fn named(name: &'a Name, arity: u8) -> Self {
        Self {
            name,
            arity,
            retry: 0,
        }
    }

    fn is_named(&self) -> Option<(Name, Arity)> {
        (self.retry == 0).then(|| (self.name.clone(), self.arity))
    }

    fn retry(self) -> Self {
        let Self { name, arity, retry } = self;
        Self {
            name,
            arity,
            retry: retry + 1,
        }
    }

    fn as_owned(&self) -> OwnedLabel {
        let Label { name, arity, retry } = *self;
        OwnedLabel {
            name: Rc::new(name.clone()),
            arity,
            retry,
        }
    }
}

impl<'a> fmt::Display for Label<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { name, arity, retry } = *self;

        match retry.checked_sub(1) {
            Some(retry) => write!(f, "retry({})", Self { name, arity, retry }),
            None => write!(f, "{name}/{arity}"),
        }
    }
}

pub struct OwnedLabel {
    name: Rc<Name>,
    arity: Arity,
    retry: usize,
}

impl OwnedLabel {
    fn named((name, arity): (Name, Arity)) -> Self {
        Self {
            name: Rc::new(name),
            arity,
            retry: 0,
        }
    }
}

impl fmt::Display for OwnedLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let OwnedLabel {
            ref name,
            arity,
            retry,
        } = *self;
        Label { name, arity, retry }.fmt(f)
    }
}

trait TokenHandler<Mode: CompilationMode> {
    fn token_xa(&mut self, xn: Xn, ai: Ai);
    fn token_va(&mut self, vn: Mode::Variable, ai: Ai);

    fn token_sa(&mut self, name: &Name, n: u8, ai: Ai);
    fn token_sx(&mut self, name: &Name, n: u8, xn: Xn) {
        self.token_sa(name, n, xn.into())
    }

    fn token_la(&mut self, ai: Ai);
    fn token_lx(&mut self, xn: Xn) {
        self.token_la(xn.into())
    }

    fn token_ca(&mut self, name: &Name, ai: Ai);
    fn token_ia(&mut self, i: &BigInt, ai: Ai);
    fn token_void_a(&mut self, ai: Ai);

    fn token_x(&mut self, xn: Xn);
    fn token_v(&mut self, vn: Mode::Variable);

    fn token_c(&mut self, name: &Name);
    fn token_i(&mut self, i: &BigInt);
    fn token_void(&mut self);
}

trait QueryTokenHandler<'i, Mode: IsQueryMode>: TokenHandler<Mode> {
    fn token_call(&mut self, label: CallName<Label<'i>>);
}

struct KnownVariables<Mode: CompilationMode>(HashSet<Mode::Variable>);

impl<Mode: CompilationMode> KnownVariables<Mode> {
    fn new() -> Self {
        Self(HashSet::new())
    }

    fn insert<V: fmt::Debug>(&mut self, vn: V) -> bool
    where
        Mode::Variable: From<V>,
    {
        self.0.insert(vn.into())
    }
}

impl KnownVariables<RuleGoalMode> {
    fn with_permanent_variables<I>(permanent_variables: I) -> Self
    where
        I: IntoIterator,
        <I as IntoIterator>::Item: Borrow<Yn>,
    {
        Self(
            permanent_variables
                .into_iter()
                .map(|variable| Vn::from(*variable.borrow()))
                .collect(),
        )
    }
}

#[derive(Debug)]
pub enum Labelled<'a, I> {
    Instruction(I),
    Label(Label<'a>),
    Call(Label<'a>),
    Execute(Label<'a>),
    TryMeElse(Label<'a>),
    RetryMeElse(Label<'a>),
}

impl<'a, I: fmt::Display> fmt::Display for Labelled<'a, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Labelled::Instruction(i) => write!(f, "{i}"),
            Labelled::Label(label) => write!(f, "label({label})"),
            Labelled::Call(label) => write!(f, "call({label})"),
            Labelled::Execute(label) => write!(f, "execute({label})"),
            Labelled::TryMeElse(label) => write!(f, "try_me_else({label})"),
            Labelled::RetryMeElse(label) => write!(f, "retry_me_else({label})"),
        }
    }
}

enum AssemblyErrorType {
    LabelNotFound {
        label: OwnedLabel,
    },
    LabelAlreadyDeclared {
        first: OwnedLabel,
        second: OwnedLabel,
    },
}

struct AssemblyError {
    error_type: AssemblyErrorType,
    program_info: ProgramInfo,
}

impl AssemblyError {
    fn print_report(self) {
        use chumsky::Span;

        match self.error_type {
            AssemblyErrorType::LabelNotFound { label } => {
                let message = format!("{label} not found");
                ariadne::Report::build(
                    ariadne::ReportKind::Error,
                    label.name.span().context(),
                    label.name.span().start(),
                )
                .with_message(&message)
                .with_label(
                    ariadne::Label::new(label.name.span().clone())
                        .with_message(message)
                        .with_color(ariadne::Color::Red),
                )
                .finish()
                .eprint(self.program_info.as_source_cache())
                .unwrap();

                let similar_labels = self
                    .program_info
                    .labels
                    .keys()
                    .filter_map(|similar_label @ (name, arity)| {
                        let name_matches = name.as_ref() == label.name.as_ref().as_ref();
                        let names_are_similar =
                            strsim::jaro_winkler(name.as_ref(), label.name.as_ref().as_ref()) > 0.8;
                        let arity_matches = *arity == label.arity;

                        (name_matches || (names_are_similar && arity_matches))
                            .then(|| ((name.span().context(), name.span().start()), similar_label))
                    })
                    .collect::<BTreeMap<_, _>>();

                for (name, arity) in similar_labels.values() {
                    let message = format!("found {name}/{arity}");

                    ariadne::Report::build(
                        ariadne::ReportKind::Advice,
                        name.span().context().clone(),
                        name.span().start(),
                    )
                    .with_message(&message)
                    .with_label(
                        ariadne::Label::new(name.span().clone())
                            .with_message(message)
                            .with_color(ariadne::Color::Green),
                    )
                    .finish()
                    .eprint(self.program_info.as_source_cache())
                    .unwrap();
                }
            }
            AssemblyErrorType::LabelAlreadyDeclared { first, second } => {
                ariadne::Report::build(
                    ariadne::ReportKind::Error,
                    second.name.span().context(),
                    second.name.span().start(),
                )
                .with_label(
                    ariadne::Label::new(second.name.span().clone())
                        .with_message(format!("{second} already declared"))
                        .with_color(ariadne::Color::Red),
                )
                .finish()
                .eprint(self.program_info.as_source_cache())
                .unwrap();

                ariadne::Report::build(
                    ariadne::ReportKind::Advice,
                    first.name.span().context(),
                    first.name.span().start(),
                )
                .with_label(
                    ariadne::Label::new(first.name.span().clone())
                        .with_message(format!("{first} first declared here"))
                        .with_color(ariadne::Color::Green),
                )
                .finish()
                .eprint(self.program_info.as_source_cache())
                .unwrap();
            }
        }
    }
}

struct LabelSet<'a> {
    program_info: &'a mut ProgramInfo,
    new_labels: HashMap<Label<'a>, u16>,
}

impl<'a> LabelSet<'a> {
    fn new(program_info: &'a mut ProgramInfo) -> Self {
        Self {
            program_info,
            new_labels: HashMap::new(),
        }
    }

    fn insert(&mut self, label: Label<'a>, pc: u16) -> Result<(), AssemblyErrorType> {
        use std::collections::hash_map::Entry;

        if let Some(name) = label.is_named() {
            match self.program_info.labels.entry(name) {
                Entry::Vacant(entry) => {
                    entry.insert(pc);
                }
                Entry::Occupied(entry) => {
                    return Err(AssemblyErrorType::LabelAlreadyDeclared {
                        first: OwnedLabel::named(entry.key().clone()),
                        second: label.as_owned(),
                    });
                }
            }
        }

        match self.new_labels.entry(label) {
            Entry::Vacant(entry) => {
                entry.insert(pc);
            }
            Entry::Occupied(entry) => {
                return Err(AssemblyErrorType::LabelAlreadyDeclared {
                    first: entry.key().as_owned(),
                    second: label.as_owned(),
                });
            }
        }

        Ok(())
    }

    fn get(&self, label: Label) -> Result<u16, AssemblyErrorType> {
        self.new_labels
            .get(&label)
            .copied()
            .or_else(|| {
                label
                    .is_named()
                    .and_then(|name| self.program_info.labels.get(&name).copied())
            })
            .ok_or_else(|| AssemblyErrorType::LabelNotFound {
                label: label.as_owned(),
            })
    }
}

struct InstructionList<'a>(Vec<Labelled<'a, Instruction>>);

impl<'a> InstructionList<'a> {
    fn new() -> Self {
        Self(Vec::new())
    }

    fn with_system_calls(system_calls: &'a [(Name, u8)]) -> Self {
        Self(
            (0..)
                .zip(system_calls.iter())
                .flat_map(|(i, (name, arity))| {
                    [
                        Labelled::Label(Label::named(name, *arity)),
                        Labelled::Instruction(Instruction::SystemCall { i }),
                        Labelled::Instruction(Instruction::Proceed),
                    ]
                })
                .collect(),
        )
    }

    fn push(&mut self, instruction: Instruction) {
        if let (
            Some((Labelled::Instruction(Instruction::PutVoid { n, .. }), _)),
            Instruction::PutVoid { n: increment, .. },
        )
        | (
            Some((Labelled::Instruction(Instruction::SetVoid { n, .. }), _)),
            Instruction::SetVoid { n: increment, .. },
        )
        | (
            Some((Labelled::Instruction(Instruction::UnifyVoid { n, .. }), _)),
            Instruction::UnifyVoid { n: increment, .. },
        ) = (self.0.split_last_mut(), &instruction)
        {
            *n += *increment;
        } else {
            self.0.push(Labelled::Instruction(instruction))
        }
    }

    fn push_label(&mut self, label: Label<'a>) {
        self.0.push(Labelled::Label(label));
    }

    fn push_call(&mut self, label: CallName<Label<'a>>) {
        self.0.push(match label {
            CallName::Named(label) => Labelled::Call(label),
            CallName::True => Labelled::Instruction(Instruction::True),
            CallName::Fail => Labelled::Instruction(Instruction::Fail),
            CallName::Is => Labelled::Instruction(Instruction::Is),
        })
    }

    fn push_try_me_else(&mut self, label: Label<'a>) {
        self.0.push(Labelled::TryMeElse(label))
    }

    fn push_retry_me_else(&mut self, label: Label<'a>) {
        self.0.push(Labelled::RetryMeElse(label))
    }

    fn last_call_optimisation(mut self) -> Self {
        // Then add proceeds after deallocates where the last goal isn't a call, e.g. true or system calls
        let mut i = 0;
        while i < self.0.len() {
            if let Some(
                [Labelled::Instruction(
                    Instruction::NeckCut
                    | Instruction::Cut { .. }
                    | Instruction::Is
                    | Instruction::True
                    | Instruction::SystemCall { .. },
                ), Labelled::Instruction(Instruction::Deallocate), ..],
            ) = self.0.get(i..)
            {
                self.0
                    .insert(i + 2, Labelled::Instruction(Instruction::Proceed))
            }

            i += 1
        }

        // Then convert Call-Decallocate to Deallocate-Execute
        let mut i = self.0.iter_mut().peekable();

        while let Some(first) = i.next() {
            if let Labelled::Call(label) = first {
                if let Some(second) = i.next_if(|second| {
                    matches!(second, Labelled::Instruction(Instruction::Deallocate))
                }) {
                    *second = Labelled::Execute(*label);
                    *first = Labelled::Instruction(Instruction::Deallocate);
                }
            }
        }

        self
    }

    fn assemble_with_program_info(
        self,
        mut program_info: ProgramInfo,
    ) -> Result<(ProgramInfo, Vec<[u8; 4]>), AssemblyError> {
        match self.assemble(&mut program_info) {
            Ok(instructions) => Ok((program_info, instructions)),
            Err(error_type) => Err(AssemblyError {
                error_type,
                program_info,
            }),
        }
    }

    fn assemble(self, program_info: &mut ProgramInfo) -> Result<Vec<[u8; 4]>, AssemblyErrorType> {
        enum AB<A, B> {
            A(A),
            B(B),
        }

        impl<A: Iterator, B: Iterator<Item = A::Item>> Iterator for AB<A, B> {
            type Item = A::Item;

            fn next(&mut self) -> Option<Self::Item> {
                match self {
                    AB::A(a) => a.next(),
                    AB::B(b) => b.next(),
                }
            }
        }

        fn just<A, I>(instruction: I) -> AB<A, impl Iterator<Item = I>> {
            AB::B(std::iter::once(instruction))
        }

        let instructions = self
            .0
            .into_iter()
            .flat_map(|instruction| match instruction {
                Labelled::Instruction(instruction) => {
                    AB::A(instruction.serialize().map(Labelled::Instruction))
                }
                Labelled::Label(label) => just(Labelled::Label(label)),
                Labelled::Call(label) => just(Labelled::Call(label)),
                Labelled::Execute(label) => just(Labelled::Execute(label)),
                Labelled::TryMeElse(label) => just(Labelled::TryMeElse(label)),
                Labelled::RetryMeElse(label) => just(Labelled::RetryMeElse(label)),
            })
            .collect::<Vec<_>>();

        let mut labels = LabelSet::new(program_info);

        instructions.iter().try_fold(0, |pc, instruction| {
            Ok(match instruction {
                Labelled::Label(label) => {
                    labels.insert(*label, pc)?;
                    pc
                }
                Labelled::Instruction(_)
                | Labelled::Call(_)
                | Labelled::Execute(_)
                | Labelled::TryMeElse(_)
                | Labelled::RetryMeElse(_) => pc + 1,
            })
        })?;

        fn assert_single<I: Iterator>(mut i: I) -> I::Item {
            let v = i.next().unwrap();
            assert!(i.next().is_none());
            v
        }

        let instructions = instructions
            .into_iter()
            .map(|instruction| {
                Ok(Some(assert_single(
                    match instruction {
                        Labelled::Instruction(instruction) => return Ok(Some(instruction)),
                        Labelled::Label(_) => return Ok(None),
                        Labelled::Call(label) => Instruction::Call {
                            p: labels.get(label)?,
                            n: label.arity,
                        },
                        Labelled::Execute(label) => Instruction::Execute {
                            p: labels.get(label)?,
                            n: label.arity,
                        },
                        Labelled::TryMeElse(label) => Instruction::TryMeElse {
                            p: labels.get(label)?,
                        },
                        Labelled::RetryMeElse(label) => Instruction::RetryMeElse {
                            p: labels.get(label)?,
                        },
                    }
                    .serialize(),
                )))
            })
            .filter_map(Result::transpose)
            .collect::<Result<_, _>>()?;

        Ok(instructions)
    }
}

enum IntegerType {
    Short(ShortInteger),
    Long(LongInteger),
}

impl<'a> From<&'a BigInt> for IntegerType {
    fn from(i: &'a BigInt) -> Self {
        i.try_into().map_or_else(
            |_| IntegerType::Long(LongInteger::from(i)),
            IntegerType::Short,
        )
    }
}

struct Assembler<'a, 'i, Mode: CompilationMode> {
    program_info: &'a mut ProgramInfo,
    known_variables: KnownVariables<Mode>,
    instructions: &'a mut InstructionList<'i>,
}

impl<'a, 'i, Mode: CompilationMode> Assembler<'a, 'i, Mode> {
    fn new(program_info: &'a mut ProgramInfo, instructions: &'a mut InstructionList<'i>) -> Self {
        Self {
            program_info,
            known_variables: KnownVariables::new(),
            instructions,
        }
    }
}

impl<'a, 'i> Assembler<'a, 'i, ProgramMode> {
    fn xa(&mut self, xn: Xn, ai: Ai) {
        self.instructions.push(if self.known_variables.insert(xn) {
            Instruction::GetVariableXn { xn, ai }
        } else {
            Instruction::GetValueXn { xn, ai }
        })
    }

    fn ya(&mut self, yn: Yn, ai: Ai) {
        self.instructions.push(if self.known_variables.insert(yn) {
            Instruction::GetVariableYn { yn, ai }
        } else {
            Instruction::GetValueYn { yn, ai }
        })
    }

    fn x(&mut self, xn: Xn) {
        self.instructions.push(if self.known_variables.insert(xn) {
            Instruction::UnifyVariableXn { xn }
        } else {
            Instruction::UnifyValueXn { xn }
        })
    }

    fn y(&mut self, yn: Yn) {
        self.instructions.push(if self.known_variables.insert(yn) {
            Instruction::UnifyVariableYn { yn }
        } else {
            Instruction::UnifyValueYn { yn }
        });
    }

    fn proceed(&mut self) {
        self.instructions.push(Instruction::Proceed)
    }

    fn allocate(&mut self, n: Arity) {
        self.instructions.push(Instruction::Allocate { n });
    }

    fn deallocate(&mut self) {
        self.instructions.push(Instruction::Deallocate);
    }

    fn trim(&mut self, n: Arity) {
        if n > 0 {
            self.instructions.push(Instruction::Trim { n });
        }
    }

    fn neck_cut(&mut self) {
        self.instructions.push(Instruction::NeckCut);
    }

    fn get_level(&mut self, yn: Yn) {
        self.instructions.push(Instruction::GetLevel { yn });
    }

    fn cut(&mut self, yn: Yn) {
        self.instructions.push(Instruction::Cut { yn });
    }
}

impl<'a, 'i> TokenHandler<ProgramMode> for Assembler<'a, 'i, ProgramMode> {
    fn token_xa(&mut self, xn: Xn, ai: Ai) {
        self.xa(xn, ai)
    }

    fn token_va(&mut self, vn: Vn, ai: Ai) {
        match vn {
            Vn::Xn { xn } => self.xa(xn, ai),
            Vn::Yn { yn } => self.ya(yn, ai),
        }
    }

    fn token_sa(&mut self, name: &Name, n: u8, ai: Ai) {
        self.known_variables.insert(ai);
        let f = self.program_info.functors.get(name.into());
        self.instructions
            .push(Instruction::GetStructure { f, n, ai })
    }

    fn token_la(&mut self, ai: Ai) {
        self.known_variables.insert(ai);
        self.instructions.push(Instruction::GetList { ai })
    }

    fn token_ca(&mut self, name: &Name, ai: Ai) {
        self.known_variables.insert(ai);
        let c = self.program_info.functors.get(name.into());
        self.instructions.push(Instruction::GetConstant { c, ai })
    }

    fn token_ia(&mut self, i: &BigInt, ai: Ai) {
        match IntegerType::from(i) {
            IntegerType::Short(i) => self
                .instructions
                .push(Instruction::GetShortInteger { i, ai }),
            IntegerType::Long(i) => self.instructions.push(Instruction::GetInteger { i, ai }),
        }
    }

    fn token_void_a(&mut self, _: Ai) {
        // There is no instruction GetVoid as it would be a no-op
    }

    fn token_x(&mut self, xn: Xn) {
        self.x(xn)
    }

    fn token_v(&mut self, vn: Vn) {
        match vn {
            Vn::Xn { xn } => self.x(xn),
            Vn::Yn { yn } => self.y(yn),
        }
    }

    fn token_c(&mut self, name: &Name) {
        let c = self.program_info.functors.get(name.into());
        self.instructions.push(Instruction::UnifyConstant { c })
    }

    fn token_i(&mut self, i: &BigInt) {
        match IntegerType::from(i) {
            IntegerType::Short(i) => self.instructions.push(Instruction::UnifyShortInteger { i }),
            IntegerType::Long(i) => self.instructions.push(Instruction::UnifyInteger { i }),
        }
    }

    fn token_void(&mut self) {
        self.instructions.push(Instruction::UnifyVoid { n: 1 });
    }
}

impl<'a, 'i> Assembler<'a, 'i, RuleGoalMode> {
    fn with_permanent_variables<I>(
        program_info: &'a mut ProgramInfo,
        permanent_variables: I,
        instructions: &'a mut InstructionList<'i>,
    ) -> Self
    where
        I: IntoIterator,
        <I as IntoIterator>::Item: Borrow<Yn>,
    {
        Self {
            program_info,
            known_variables: KnownVariables::with_permanent_variables(permanent_variables),
            instructions,
        }
    }

    fn xa(&mut self, xn: Xn, ai: Ai) {
        self.instructions.push(if self.known_variables.insert(xn) {
            Instruction::PutVariableXn { xn, ai }
        } else {
            Instruction::PutValueXn { xn, ai }
        })
    }

    fn ya(&mut self, yn: Yn, ai: Ai) {
        self.instructions.push(if self.known_variables.insert(yn) {
            Instruction::PutVariableYn { yn, ai }
        } else {
            Instruction::PutValueYn { yn, ai }
        })
    }

    fn x(&mut self, xn: Xn) {
        self.instructions.push(if self.known_variables.insert(xn) {
            Instruction::SetVariableXn { xn }
        } else {
            Instruction::SetValueXn { xn }
        })
    }

    fn y(&mut self, yn: Yn) {
        self.instructions.push(if self.known_variables.insert(yn) {
            Instruction::SetVariableYn { yn }
        } else {
            Instruction::SetValueYn { yn }
        })
    }
}

impl<'a, 'i> TokenHandler<RuleGoalMode> for Assembler<'a, 'i, RuleGoalMode> {
    fn token_xa(&mut self, xn: Xn, ai: Ai) {
        self.xa(xn, ai)
    }

    fn token_va(&mut self, vn: Vn, ai: Ai) {
        match vn {
            Vn::Xn { xn } => self.xa(xn, ai),
            Vn::Yn { yn } => self.ya(yn, ai),
        }
    }

    fn token_sa(&mut self, name: &Name, n: u8, ai: Ai) {
        self.known_variables.insert(ai);
        let f = self.program_info.functors.get(name.into());
        self.instructions
            .push(Instruction::PutStructure { f, n, ai })
    }

    fn token_la(&mut self, ai: Ai) {
        self.known_variables.insert(ai);
        self.instructions.push(Instruction::PutList { ai })
    }

    fn token_ca(&mut self, name: &Name, ai: Ai) {
        self.known_variables.insert(ai);
        let c = self.program_info.functors.get(name.into());
        self.instructions.push(Instruction::PutConstant { c, ai })
    }

    fn token_ia(&mut self, i: &BigInt, ai: Ai) {
        match IntegerType::from(i) {
            IntegerType::Short(i) => self
                .instructions
                .push(Instruction::PutShortInteger { i, ai }),
            IntegerType::Long(i) => self.instructions.push(Instruction::PutInteger { i, ai }),
        }
    }

    fn token_void_a(&mut self, ai: Ai) {
        self.instructions.push(Instruction::PutVoid { n: 1, ai });
    }

    fn token_x(&mut self, xn: Xn) {
        self.x(xn)
    }

    fn token_v(&mut self, vn: Vn) {
        match vn {
            Vn::Xn { xn } => self.x(xn),
            Vn::Yn { yn } => self.y(yn),
        }
    }

    fn token_c(&mut self, name: &Name) {
        let c = self.program_info.functors.get(name.into());
        self.instructions.push(Instruction::SetConstant { c });
    }

    fn token_i(&mut self, i: &BigInt) {
        match IntegerType::from(i) {
            IntegerType::Short(i) => self.instructions.push(Instruction::SetShortInteger { i }),
            IntegerType::Long(i) => self.instructions.push(Instruction::SetInteger { i }),
        }
    }

    fn token_void(&mut self) {
        self.instructions.push(Instruction::SetVoid { n: 1 });
    }
}

impl<'a, 'i> QueryTokenHandler<'i, RuleGoalMode> for Assembler<'a, 'i, RuleGoalMode> {
    fn token_call(&mut self, label: CallName<Label<'i>>) {
        self.instructions.push_call(label)
    }
}

impl<'a, 'i> TokenHandler<QueryMode> for Assembler<'a, 'i, QueryMode> {
    fn token_xa(&mut self, xn: Xn, ai: Ai) {
        self.instructions.push(if self.known_variables.insert(xn) {
            Instruction::PutVariableXn { xn, ai }
        } else {
            Instruction::PutValueXn { xn, ai }
        })
    }

    fn token_va(&mut self, xn: Xn, ai: Ai) {
        self.token_xa(xn, ai)
    }

    fn token_sa(&mut self, name: &Name, n: u8, ai: Ai) {
        self.known_variables.insert(ai);
        let f = self.program_info.functors.get(name.into());
        self.instructions
            .push(Instruction::PutStructure { f, n, ai })
    }

    fn token_la(&mut self, ai: Ai) {
        self.known_variables.insert(ai);
        self.instructions.push(Instruction::PutList { ai })
    }

    fn token_ca(&mut self, name: &Name, ai: Ai) {
        self.known_variables.insert(ai);
        let c = self.program_info.functors.get(name.into());
        self.instructions.push(Instruction::PutConstant { c, ai })
    }

    fn token_ia(&mut self, i: &BigInt, ai: Ai) {
        match IntegerType::from(i) {
            IntegerType::Short(i) => self
                .instructions
                .push(Instruction::PutShortInteger { i, ai }),
            IntegerType::Long(i) => self.instructions.push(Instruction::PutInteger { i, ai }),
        }
    }

    fn token_void_a(&mut self, ai: Ai) {
        self.instructions.push(Instruction::PutVoid { n: 1, ai });
    }

    fn token_x(&mut self, xn: Xn) {
        self.instructions.push(if self.known_variables.insert(xn) {
            Instruction::SetVariableXn { xn }
        } else {
            Instruction::SetValueXn { xn }
        })
    }

    fn token_v(&mut self, xn: Xn) {
        self.token_x(xn)
    }

    fn token_c(&mut self, name: &Name) {
        let c = self.program_info.functors.get(name.into());
        self.instructions.push(Instruction::SetConstant { c });
    }

    fn token_i(&mut self, i: &BigInt) {
        match IntegerType::from(i) {
            IntegerType::Short(i) => self.instructions.push(Instruction::SetShortInteger { i }),
            IntegerType::Long(i) => self.instructions.push(Instruction::SetInteger { i }),
        }
    }

    fn token_void(&mut self) {
        self.instructions.push(Instruction::SetVoid { n: 1 });
    }
}

impl<'a, 'i> QueryTokenHandler<'i, QueryMode> for Assembler<'a, 'i, QueryMode> {
    fn token_call(&mut self, label: CallName<Label<'i>>) {
        self.instructions.push_call(label);
        self.instructions.push(Instruction::Proceed);
    }
}

struct AllocationTokenisationState<'a, 'i, Mode: CompilationMode> {
    assembler: Assembler<'a, 'i, Mode>,
}

impl<'a, 'i, Mode: CompilationMode> AllocationTokenisationState<'a, 'i, Mode>
where
    Assembler<'a, 'i, Mode>: TokenHandler<Mode>,
{
    fn new(program_info: &'a mut ProgramInfo, instructions: &'a mut InstructionList<'i>) -> Self {
        Self {
            assembler: Assembler::new(program_info, instructions),
        }
    }

    fn tokenize_atom_argument_allocation_list(&mut self, allocations: &[ArgumentAllocation<Mode>]) {
        for allocation in allocations {
            match *allocation {
                ArgumentAllocation::Variable { ai, vn, .. } => self.assembler.token_va(vn, ai),
                ArgumentAllocation::Structure {
                    ai,
                    name,
                    ref terms,
                } => {
                    self.assembler.token_sa(name, terms.len() as u8, ai);
                    for term in terms {
                        match *term {
                            TermAllocation::Variable { vn, .. } => self.assembler.token_v(vn),
                            TermAllocation::Structure { xn, .. }
                            | TermAllocation::List { xn, .. } => self.assembler.token_x(xn),
                            TermAllocation::Constant { name } => self.assembler.token_c(name),
                            TermAllocation::Integer { i } => self.assembler.token_i(i),
                            TermAllocation::Void => self.assembler.token_void(),
                        }
                    }
                }
                ArgumentAllocation::List { ai, ref terms } => {
                    self.assembler.token_la(ai);
                    for term in terms {
                        match *term {
                            TermAllocation::Variable { vn, .. } => self.assembler.token_v(vn),
                            TermAllocation::Structure { xn, .. }
                            | TermAllocation::List { xn, .. } => self.assembler.token_x(xn),
                            TermAllocation::Constant { name } => self.assembler.token_c(name),
                            TermAllocation::Integer { i } => self.assembler.token_i(i),
                            TermAllocation::Void => self.assembler.token_void(),
                        }
                    }
                }
                ArgumentAllocation::Constant { ai, name } => {
                    self.assembler.token_ca(name, ai);
                }
                ArgumentAllocation::Integer { ai, i } => {
                    self.assembler.token_ia(i, ai);
                }
                ArgumentAllocation::Void { ai } => self.assembler.token_void_a(ai),
            }
        }
    }
}

enum GoalAllocation<'a> {
    Goal {
        name: CallName<&'a Name>,
        allocations: Vec<ArgumentAllocation<'a, RuleGoalMode>>,
    },
    NeckCut,
    Cut,
}

impl<'a, 'i> AllocationTokenisationState<'a, 'i, ProgramMode> {
    fn goals<'b, I>(
        &'b mut self,
        already_declared_permanent_variables: I,
    ) -> AllocationTokenisationState<'b, 'i, RuleGoalMode>
    where
        I: IntoIterator,
        <I as IntoIterator>::Item: Borrow<Yn>,
    {
        AllocationTokenisationState {
            assembler: Assembler::with_permanent_variables(
                self.assembler.program_info,
                already_declared_permanent_variables,
                self.assembler.instructions,
            ),
        }
    }

    fn tokenize_program_subterm(&mut self, term: &TermAllocation<ProgramMode>) {
        match term {
            TermAllocation::Variable { .. } => (),
            &TermAllocation::Structure {
                xn,
                name,
                ref terms,
            } => {
                self.assembler.token_sx(name, terms.len() as u8, xn);

                for term in terms {
                    match *term {
                        TermAllocation::Variable { vn, .. } => self.assembler.token_v(vn),
                        TermAllocation::Structure { xn, .. } | TermAllocation::List { xn, .. } => {
                            self.assembler.token_x(xn)
                        }
                        TermAllocation::Constant { name } => self.assembler.token_c(name),
                        TermAllocation::Integer { i } => self.assembler.token_i(i),
                        TermAllocation::Void => self.assembler.token_void(),
                    }
                }

                for term in terms {
                    self.tokenize_program_subterm(term);
                }
            }
            &TermAllocation::List { xn, ref terms } => {
                self.assembler.token_lx(xn);

                for term in terms {
                    match *term {
                        TermAllocation::Variable { vn, .. } => self.assembler.token_v(vn),
                        TermAllocation::Structure { xn, .. } | TermAllocation::List { xn, .. } => {
                            self.assembler.token_x(xn)
                        }
                        TermAllocation::Constant { name } => self.assembler.token_c(name),
                        TermAllocation::Integer { i } => self.assembler.token_i(i),
                        TermAllocation::Void => self.assembler.token_void(),
                    }
                }

                for term in terms {
                    self.tokenize_program_subterm(term);
                }
            }
            TermAllocation::Constant { .. } => (),
            TermAllocation::Integer { .. } => (),
            TermAllocation::Void => (),
        }
    }

    fn tokenize_fact_allocation(mut self, allocations: Vec<ArgumentAllocation<ProgramMode>>) {
        self.tokenize_atom_argument_allocation_list(&allocations);

        for allocation in &allocations {
            match allocation {
                ArgumentAllocation::Variable { .. }
                | ArgumentAllocation::Constant { .. }
                | ArgumentAllocation::Integer { .. }
                | ArgumentAllocation::Void { .. } => (),
                ArgumentAllocation::Structure { terms, .. }
                | ArgumentAllocation::List { terms, .. } => {
                    for term in terms {
                        self.tokenize_program_subterm(term);
                    }
                }
            }
        }

        self.assembler.proceed();
    }

    fn tokenize_rule_allocation(
        mut self,
        head_allocations: Vec<ArgumentAllocation<'i, ProgramMode>>,
        body_allocations: Vec<GoalAllocation<'i>>,
        PermanentVariablesAllocation {
            permanent_variables,
            already_declared_permanent_variables,
            trimmed_permanent_variables,
        }: PermanentVariablesAllocation,
    ) {
        let permanent_variables = permanent_variables
            .iter()
            .copied()
            .zip((0_u8..).map(|yn| Yn { yn }))
            .collect::<HashMap<_, _>>();

        self.assembler.allocate(permanent_variables.len() as u8);

        if let Some(&yn) = permanent_variables.get(&VariableName::Cut) {
            self.assembler.get_level(yn);
        }

        self.tokenize_atom_argument_allocation_list(&head_allocations);

        for allocation in head_allocations {
            match allocation {
                ArgumentAllocation::Variable { .. }
                | ArgumentAllocation::Constant { .. }
                | ArgumentAllocation::Integer { .. }
                | ArgumentAllocation::Void { .. } => (),
                ArgumentAllocation::Structure { terms, .. }
                | ArgumentAllocation::List { terms, .. } => {
                    for term in terms {
                        self.tokenize_program_subterm(&term);
                    }
                }
            }
        }

        for (
            body_allocation,
            (already_declared_permanent_variables, trimmed_permanent_variables),
        ) in body_allocations.into_iter().zip(
            already_declared_permanent_variables
                .into_iter()
                .zip(&trimmed_permanent_variables),
        ) {
            match body_allocation {
                GoalAllocation::Goal { name, allocations } => {
                    let already_declared_permanent_variables = already_declared_permanent_variables
                        .iter()
                        .map(|permanent_variable| {
                            permanent_variables.get(permanent_variable).unwrap()
                        });

                    self.goals(already_declared_permanent_variables)
                        .tokenize_query_allocations(name, &allocations);

                    self.assembler.trim(*trimmed_permanent_variables);
                }
                GoalAllocation::NeckCut => self.assembler.neck_cut(),
                GoalAllocation::Cut => self
                    .assembler
                    .cut(*permanent_variables.get(&VariableName::Cut).unwrap()),
            }
        }

        self.assembler.deallocate();
    }
}

impl<'a, 'i, Mode: IsQueryMode> AllocationTokenisationState<'a, 'i, Mode>
where
    Assembler<'a, 'i, Mode>: QueryTokenHandler<'i, Mode>,
{
    fn tokenize_query_subterm(&mut self, term: &TermAllocation<Mode>) {
        match term {
            TermAllocation::Variable { .. } => (),
            &TermAllocation::Structure {
                xn,
                name,
                ref terms,
            } => {
                for term in terms {
                    self.tokenize_query_subterm(term);
                }

                self.assembler.token_sx(name, terms.len() as u8, xn);

                for term in terms {
                    match *term {
                        TermAllocation::Variable { vn, .. } => self.assembler.token_v(vn),
                        TermAllocation::Structure { xn, .. } | TermAllocation::List { xn, .. } => {
                            self.assembler.token_x(xn)
                        }
                        TermAllocation::Constant { name } => self.assembler.token_c(name),
                        TermAllocation::Integer { i } => self.assembler.token_i(i),
                        TermAllocation::Void => self.assembler.token_void(),
                    }
                }
            }

            &TermAllocation::List { xn, ref terms } => {
                for term in terms {
                    self.tokenize_query_subterm(term);
                }

                self.assembler.token_lx(xn);

                for term in terms {
                    match *term {
                        TermAllocation::Variable { vn, .. } => self.assembler.token_v(vn),
                        TermAllocation::Structure { xn, .. } | TermAllocation::List { xn, .. } => {
                            self.assembler.token_x(xn)
                        }
                        TermAllocation::Constant { name } => self.assembler.token_c(name),
                        TermAllocation::Integer { i } => self.assembler.token_i(i),
                        TermAllocation::Void => self.assembler.token_void(),
                    }
                }
            }
            TermAllocation::Constant { .. } => (),
            TermAllocation::Integer { .. } => (),
            TermAllocation::Void => (),
        }
    }

    fn tokenize_query_allocations(
        mut self,
        name: CallName<&'i Name>,
        allocations: &Vec<ArgumentAllocation<'a, Mode>>,
    ) {
        for allocation in allocations {
            match allocation {
                ArgumentAllocation::Variable { .. }
                | ArgumentAllocation::Constant { .. }
                | ArgumentAllocation::Integer { .. }
                | ArgumentAllocation::Void { .. } => (),
                ArgumentAllocation::Structure { terms, .. }
                | ArgumentAllocation::List { terms, .. } => {
                    for term in terms {
                        self.tokenize_query_subterm(term);
                    }
                }
            }
        }

        self.tokenize_atom_argument_allocation_list(allocations);

        self.assembler
            .token_call(name.map_name(|name| Label::named(name, allocations.len() as u8)))
    }
}

fn compile_program_definition<'i>(
    Definition { head, body }: &'i Definition,
    program_info: &mut ProgramInfo,
    instructions: &mut InstructionList<'i>,
) {
    if body.is_empty() {
        let allocations = RegisterAllocationState::program().allocate_argument_registers(head);

        AllocationTokenisationState::new(program_info, instructions)
            .tokenize_fact_allocation(allocations);
    } else {
        // allocate_permanent_variables
        let permanent_variable_allocation = PermanentVariablesAllocation::new(head, body);

        // allocate_atom_registers
        let head_allocations = RegisterAllocationState::with_permanent_variables(
            &permanent_variable_allocation.permanent_variables,
        )
        .allocate_argument_registers(head);

        // allocate_goals_registers
        let body_allocations = body
            .iter()
            .map(|goal| match goal {
                Goal::Named { name, terms } => GoalAllocation::Goal {
                    name: name.as_ref(),
                    allocations: RegisterAllocationState::with_permanent_variables(
                        &permanent_variable_allocation.permanent_variables,
                    )
                    .allocate_argument_registers(terms),
                },
                Goal::NeckCut => GoalAllocation::NeckCut,
                Goal::Cut => GoalAllocation::Cut,
            })
            .collect();

        AllocationTokenisationState::new(program_info, instructions).tokenize_rule_allocation(
            head_allocations,
            body_allocations,
            permanent_variable_allocation,
        );
    }
}

#[derive(Debug, thiserror::Error)]
#[error("Failed to compile program")]
pub struct CompileProgramError;

pub fn compile_program(
    system_calls: Vec<(String, u8)>,
    path: PathBuf,
) -> Result<(ProgramInfo, Vec<[u8; 4]>), CompileProgramError> {
    let path = Rc::<Path>::from(path);

    let program_source = std::fs::read_to_string(&path).map_err(|error| {
        eprintln!("Failed to open {}: {}", path.display(), error);

        CompileProgramError
    })?;

    let mut system_call_source = String::new();

    let system_calls = system_calls
        .into_iter()
        .map(|(system_call_name, arity)| {
            use std::fmt::Write;

            struct SystemCallArguments(Arity);

            impl fmt::Display for SystemCallArguments {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    let Self(arity) = *self;

                    if arity == 0 {
                        Ok(())
                    } else {
                        write!(f, "(_")?;

                        for _ in 1..arity {
                            write!(f, ",_")?;
                        }

                        write!(f, ")")
                    }
                }
            }

            let start = system_call_source.chars().count();

            write!(&mut system_call_source, "{system_call_name}").unwrap();

            let end = system_call_source.chars().count();

            writeln!(
                &mut system_call_source,
                "{} :- <system call>",
                SystemCallArguments(arity)
            )
            .unwrap();

            (
                Name::new(
                    ArcStr::from(system_call_name),
                    (SourceId::SystemCall, start..end),
                ),
                arity,
            )
        })
        .collect::<Vec<_>>();

    let mut program_info = ProgramInfo::new(&system_call_source, path.clone(), &program_source);

    let program = parser::parse_program(path, &program_source).map_err(|errors| {
        for error in errors {
            parser::print_parse_error_report(error, &program_info);
        }

        CompileProgramError
    })?;

    let mut instructions = InstructionList::with_system_calls(&system_calls);

    for disjunction in &program.definitions {
        match disjunction {
            Disjunction::Single {
                name,
                arity,
                definition,
            } => {
                instructions.push_label(Label::named(name, *arity));

                compile_program_definition(definition, &mut program_info, &mut instructions);
            }
            Disjunction::Multiple {
                name,
                arity,
                first_definition,
                middle_definitions,
                last_definition,
            } => {
                let label = Label::named(name, *arity);
                instructions.push_label(label);

                let label = label.retry();
                instructions.push_try_me_else(label);

                compile_program_definition(first_definition, &mut program_info, &mut instructions);

                let label = middle_definitions.iter().fold(label, |label, definition| {
                    instructions.push_label(label);

                    let label = label.retry();
                    instructions.push_retry_me_else(label);

                    compile_program_definition(definition, &mut program_info, &mut instructions);

                    label
                });

                instructions.push_label(label);
                instructions.push(Instruction::TrustMe);

                compile_program_definition(last_definition, &mut program_info, &mut instructions);
            }
        }
    }

    instructions
        .last_call_optimisation()
        .assemble_with_program_info(program_info)
        .map_err(|label_not_found| {
            label_not_found.print_report();

            CompileProgramError
        })
}

#[derive(Debug, thiserror::Error)]
#[error("Failed to compile query")]
pub struct CompileQueryError;

pub fn compile_query(
    query_source: &str,
    program_info: &ProgramInfo,
) -> Result<(ProgramInfo, Query, Vec<[u8; 4]>), CompileQueryError> {
    let mut program_info = program_info.with_query_source(query_source);

    let query = parser::parse_query(query_source).map_err(|errors| {
        for error in errors {
            parser::print_parse_error_report(error, &program_info);
        }

        CompileQueryError
    })?;

    let allocations = RegisterAllocationState::query().allocate_argument_registers(&query.terms);

    let mut instructions = InstructionList::new();
    AllocationTokenisationState::new(&mut program_info, &mut instructions)
        .tokenize_query_allocations(query.name.as_ref(), &allocations);

    let (program_info, words) = instructions
        .assemble_with_program_info(program_info)
        .map_err(|assembly_error| {
            assembly_error.print_report();

            CompileQueryError
        })?;

    Ok((program_info, query, words))
}
