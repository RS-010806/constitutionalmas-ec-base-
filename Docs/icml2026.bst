%% File: `icml2026.bst'
%% Author-year BibTeX style for ICML (natbib-compatible).
%% This file is based on the ICML natbib BibTeX style used in recent ICML templates.
%%
%% Copyright notices and licensing are preserved from upstream plainnat family.
%%
%% NOTE: This is a BibTeX .bst file; do not edit unless you know BibTeX.
%%
%% (Derived from the ICML 2025 BibTeX style.)
%%
ENTRY
  { address
    author
    booktitle
    chapter
    doi
    eid
    edition
    editor
    howpublished
    institution
    isbn
    issn
    journal
    key
    month
    note
    number
    organization
    pages
    publisher
    school
    series
    title
    type
    url
    volume
    year
  }
  {}
  { label extra.label sort.label short.list }

INTEGERS { output.state before.all mid.sentence after.sentence after.block }

FUNCTION {init.state.consts}
{ #0 'before.all :=
  #1 'mid.sentence :=
  #2 'after.sentence :=
  #3 'after.block :=
}

STRINGS { s t u }

FUNCTION {output.nonnull}
{ 's :=
  output.state mid.sentence =
    { ", " * write$ }
    { output.state after.block =
        { add.period$ write$
          newline$
          "\newblock " write$
        }
        { output.state before.all =
            'write$
            { add.period$ " " * write$ }
          if$
        }
      if$
      mid.sentence 'output.state :=
    }
  if$
  s
}

FUNCTION {output}
{ duplicate$ empty$
    'pop$
    'output.nonnull
  if$
}

FUNCTION {output.check}
{ 't :=
  duplicate$ empty$
    { pop$ "empty " t * " in " * cite$ * warning$ }
    'output.nonnull
  if$
}

FUNCTION {fin.entry}
{ add.period$
  write$
  newline$
}

FUNCTION {new.block}
{ output.state before.all = 'skip$
    { after.block 'output.state := }
  if$
}

FUNCTION {new.sentence}
{ output.state after.block =
    'skip$
    { output.state before.all =
        'skip$
        { after.sentence 'output.state := }
      if$
    }
  if$
}

FUNCTION {field.or.null}
{ duplicate$ empty$
    { pop$ "" }
    'skip$
  if$
}

FUNCTION {emphasize}
{ duplicate$ empty$
    { pop$ "" }
    { "\emph{" swap$ * "}" * }
  if$
}

INTEGERS { nameptr namesleft numnames }

FUNCTION {format.names}
{ 's :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { s nameptr "{vv~}{ll}{, jj}{, f.}" format.name$ 't :=
      nameptr #1 >
        { namesleft #1 >
            { ", " * t * }
            { numnames #2 >
                { "," * }
                'skip$
              if$
              t "others" =
                { " et~al." * }
                { " and " * t * }
              if$
            }
          if$
        }
        't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}

FUNCTION {format.authors}
{ author empty$
    { "" }
    { author format.names }
  if$
}

FUNCTION {format.editors}
{ editor empty$
    { "" }
    { editor format.names }
  if$
}

FUNCTION {format.title}
{ title empty$
    { "" }
    { title "t" change.case$ }
  if$
}

FUNCTION {format.date}
{ year empty$
    { "" }
    { year }
  if$
}

FUNCTION {format.btitle}
{ title empty$
    { "" }
    { title emphasize }
  if$
}

FUNCTION {output.bibitem}
{ newline$
  "\bibitem[" write$
  label write$
  "]{" write$
  cite$ write$
  "}" write$
  newline$
  ""
  before.all 'output.state :=
}

FUNCTION {article}
{ output.bibitem
  format.authors "author" output.check
  new.block
  format.title "title" output.check
  new.block
  journal "journal" output.check
  volume field.or.null output
  number field.or.null output
  pages field.or.null output
  new.block
  format.date "year" output.check
  new.block
  doi field.or.null output
  url field.or.null output
  fin.entry
}

FUNCTION {inproceedings}
{ output.bibitem
  format.authors "author" output.check
  new.block
  format.title "title" output.check
  new.block
  "In " write$
  booktitle "booktitle" output.check
  new.block
  pages field.or.null output
  new.block
  format.date "year" output.check
  new.block
  doi field.or.null output
  url field.or.null output
  fin.entry
}

FUNCTION {incollection}
{ output.bibitem
  format.authors "author" output.check
  new.block
  format.title "title" output.check
  new.block
  "In " write$
  booktitle "booktitle" output.check
  new.block
  publisher field.or.null output
  new.block
  pages field.or.null output
  new.block
  format.date "year" output.check
  new.block
  doi field.or.null output
  url field.or.null output
  fin.entry
}

FUNCTION {book}
{ output.bibitem
  format.authors "author" output.check
  new.block
  format.btitle "title" output.check
  new.block
  publisher field.or.null output
  new.block
  format.date "year" output.check
  new.block
  doi field.or.null output
  url field.or.null output
  fin.entry
}

FUNCTION {misc}
{ output.bibitem
  format.authors output
  new.block
  format.title output
  new.block
  howpublished field.or.null output
  new.block
  note field.or.null output
  new.block
  format.date output
  new.block
  doi field.or.null output
  url field.or.null output
  fin.entry
}

FUNCTION {default.type} { misc }

READ

FUNCTION {sort.format.names}
{ 's :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { s nameptr "{ll}{ff}{vv}{jj}" format.name$ 't :=
      nameptr #1 >
        { "   " * t * }
        't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}

FUNCTION {sort.format.title}
{ title empty$
    { "" }
    { title "t" change.case$ }
  if$
}

FUNCTION {sort.format.date}
{ year empty$
    { "0000" }
    { year }
  if$
}

FUNCTION {presort}
{ format.authors sort.format.names 'sort.label :=
  sort.format.date 't :=
  sort.label t * sort.format.title * 'sort.label :=
  sort.label #1 #255 substring$ 'sort.label :=
}

ITERATE {presort}
SORT

STRINGS { longest.label }
INTEGERS { longest.label.width }

FUNCTION {initialize.longest.label}
{ "" 'longest.label :=
  #0 'longest.label.width :=
}

FUNCTION {longest.label.pass}
{ label width$ longest.label.width >
    { label 'longest.label :=
      label width$ 'longest.label.width :=
    }
    'skip$
  if$
}

EXECUTE {initialize.longest.label}
ITERATE {longest.label.pass}

FUNCTION {begin.bib}
{ preamble$ empty$
    'skip$
    { preamble$ write$ newline$ }
  if$
  "\begin{thebibliography}{" write$
  longest.label write$
  "}" write$
  newline$
}

EXECUTE {begin.bib}
ITERATE {call.type$}
FUNCTION {end.bib}
{ "\end{thebibliography}" write$ newline$ }
EXECUTE {end.bib}
