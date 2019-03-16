%% $UFDate: 2017-09-21 11:24:36 +0200 -- Commit: 6bfa13a (HEAD, master) -- master$
%% nowfnt.cls v20170921
%% based on earler cls by
%% (c) 2013 Neal Parikh (Stanford University)
%%
%% Based on earlier NOW style files by S. Kumar,
%% J. Sebastian, T. Hoekwater, and B.V. Elvenkind

\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{nowfnt}
              [2017/09/21 v0.95a
               NOW document class]

\providecommand\nowfntlistofjournals{examplefnt}
\newcommand\newnowfntjournal[1]{%
  \g@addto@macro\nowfntlistofjournals{,#1}%
  }
\RequirePackage{etoolbox}

% in this cfg-file new journals can be declared
\InputIfFileExists{nowfnt-listofjournals.cfg}{}{}
\input{NOWFnT-data}

%every journal shortcut creates an option:
\@for\now@temp:=\nowfntlistofjournals\do
 {\edef\next{\noexpand\DeclareOption{\now@temp}{\noexpand\def\noexpand\nowfnt@journalkey{\now@temp}}}%
  \next}

% use 'book' as a base class
\PassOptionsToClass{letterpaper,11pt,twoside,onecolumn,final}{book}

\newif\if@nowfntbook
\newcommand\nowfnt@bookstate{0}
\newif\if@nowfntjournal
      \@nowfntjournaltrue
\newcommand\nowfnt@journalstate{1}
\newif\if@nowfntebook
\newcommand\nowfnt@ebookstate{0}
\newif\if@nowfntsample
      \@nowfntsamplefalse
\newcommand\nowfnt@samplestate{0}
\newif\if@nowfntchapter
\newcommand\nowfnt@chapterstate{0}
\newif\if@nowfntbiber
\newif\if@nowfntwrapper
\newif\if@nowfntwrapperbook
\newif\if@nowfntwrapperebook
\newif\if@nowfntchaptertwoside
\newif\if@nowfntplain

\@nowfntwrapperfalse

\DeclareOption{book}
 {%
  \@nowfntbooktrue
  \def\nowfnt@bookstate{1}%
  \@nowfntjournalfalse
  \def\nowfnt@journalstate{0}%
  }
\DeclareOption{ebook}
 {%
  \@nowfntebooktrue
  \def\nowfnt@ebookstate{1}%
  \def\nowfnt@journalstate{0}
  \@nowfntjournalfalse
 }

\DeclareOption{sample}
 {%
  \@nowfntsampletrue
  \def\nowfnt@samplestate{1}%
  \def\nowfnt@journalstate{0}
  \@nowfntjournalfalse
 }

\DeclareOption{chapter}
 {%
  \@nowfntchaptertrue
  \@nowfntchaptertwosidetrue
  \def\nowfnt@chapterstate{1}%
  \def\nowfnt@journalstate{0}
  \@nowfntjournalfalse
 }

\DeclareOption{echapter}
 {%
  \@nowfntchaptertrue
  \@nowfntchaptertwosidefalse
  \def\nowfnt@chapterstate{1}%
  \def\nowfnt@journalstate{0}
  \@nowfntjournalfalse
 }

\DeclareOption{plain}
 {%
  \@nowfntplaintrue
  \def\nowfnt@journalstate{1}
  \def\nowfnt@bookstate{0}%
  \@nowfntjournalfalse
 }
\DeclareOption{wrapper-book}
 {%
  \@nowfntwrappertrue
  \@nowfntwrapperbooktrue
  \def\nowfnt@journalstate{0}
  \def\nowfnt@bookstate{1}%
  \@nowfntjournalfalse
 }

\DeclareOption{wrapper-ebook}
 {%
  \@nowfntwrappertrue
  \@nowfntwrapperebooktrue
  \def\nowfnt@journalstate{0}
  \def\nowfnt@bookstate{1}%
  \@nowfntjournalfalse
 }


\DeclareOption{biber}  {\@nowfntbibertrue}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}

\ExecuteOptions{examplefnt,journal}
\ProcessOptions

\ifnum\numexpr \nowfnt@bookstate+ \nowfnt@ebookstate + \nowfnt@journalstate \nowfnt@samplestate= 1
\else
   \ClassWarning{nowfnt}{Unclear state. Remove the class option book or ebook or sample}{}
\fi

\ifboolexpr{bool {@nowfntbook} or bool {@nowfntwrapperbook} or bool{@nowfntchaptertwoside}}
{}
{%
 \PassOptionsToClass{openany}{book}
}

\ifbool{@nowfntplain}
{\PassOptionsToClass{oneside,notitlepage}{book}
 \LoadClass{book}}
{\LoadClass{book}}

\ifbool{@nowfntplain}
{%\let\chapter\section
 %\let\section\subsection
% \let\subsection\subsubsection
% \let\subsubsection\paragraph
% \let\paragraph\subparagraph

\newenvironment{abstract}{%
      \if@twocolumn
        \section*{\abstractname}%
      \else
        \small
        \begin{center}%
          {\bfseries \abstractname\vspace{-.5em}\vspace{\z@}}%
        \end{center}%
        \quotation
      \fi}
      {\if@twocolumn\else\endquotation\fi}
}
{}


\RequirePackage{lastpage}
\RequirePackage{multicol}
\RequirePackage{array}
\RequirePackage{trimspaces}
\RequirePackage[bottom,multiple]{footmisc}
\PassOptionsToPackage{dvipsnames}{xcolor}
\RequirePackage{pdfpages}
\RequirePackage{xr}

%some temporary registers
\newcounter{nowfnt@tempcnt}
%setup journaldata commands

\newcommand\journallibraryinfo  [1]{\def\nowfnt@journallibraryinfo{#1}}
           \journallibraryinfo{}
\newcommand\journaltitle        [1]{\def\nowfnt@journaltitle{#1}}
           \journaltitle{}
\newcommand\journalshorttitle   [1]{\def\nowfnt@journalshorttitle{#1}}
           \journalshorttitle{}
\newcommand\journaltitleprefix  [1]{\def\nowfnt@journaltitleprefix{#1}}
           \journaltitleprefix{}
\newcommand\journalurl          [1]{\def\nowfnt@journalurl{#1}}
           \journalurl{}
\newcommand\journalissnprint    [1]{\def\nowfnt@journalissnprint{#1}}
           \journalissnprint{}
\newcommand\journalissnonline   [1]{\def\nowfnt@journalissnonline{#1}}
           \journalissnonline{}
\newcommand\journaldoi          [1]{\def\nowfnt@journaldoi{#1}}
           \journaldoi{}
\newcommand\journalcopyrightowner[1]{\def\nowfnt@journalcopyrightowner{#1}}
           \journalcopyrightowner{}
\newcommand\journalaimsandscope [1]{\def\nowfnt@journalaimsandscope{#1}}
           \journalaimsandscope{}
\newcommand\journalbibstyle     [1]{%
 \ifboolexpr{ test {\ifstrequal{#1}{numeric}} or test {\ifstrequal{#1}{author}} }
  {\def\nowfnt@journalbibstyle{#1}}
  {\ClassError{nowfnt}{journalbibstyle must be numeric or author}}{}{}}
           \journalbibstyle{author}

\newcommand\nowfnt@journalcolor{.}
\newcommand\journalcolor[1]{\def\nowfnt@journalcolor{#1}}

\InputIfFileExists{\nowfnt@journalkey-journaldata}{}


\newcommand\usejournaldata[1]{%
 \ifcsdef{nowfnt@journal#1}
 {\csname nowfnt@journal#1\endcsname}
 {\ClassError{nowfnt}{The command nowfnt@journal#1 is not defined}{}{}}}

\newcommand\useissuedata[1]{%
 \ifcsdef{nowfnt@issue#1}
 {\csname nowfnt@issue#1\endcsname}
 {\ClassError{nowfnt}{The command nowfnt@issue#1 is not defined}{}{}}}

\RequirePackage{keyval}
% setup issue information with keyval syntax
\define@key{nowfnt}
  {doi}{\def\nowfnt@issuedoi{#1}}
\define@key{nowfnt}
  {firstpage}{\def\nowfnt@issuefirstpage{#1}}
\define@key{nowfnt}
  {lastpage}{\def\nowfnt@issuelastpage{#1}}
\define@key{nowfnt}
 {isbn}{\def\nowfnt@issueisbn{#1}}
\define@key{nowfnt}
 {eisbn}{\def\nowfnt@issueeisbn{#1}}
\define@key{nowfnt}
 {issue}{\def\nowfnt@issueissue{#1}}
\define@key{nowfnt}
 {copyrightyear}{\edef\nowfnt@issuecopyrightyear{#1}}
\define@key{nowfnt}
 {copyrightowner}{\def\nowfnt@issuecopyrightowner{#1}}
\define@key{nowfnt}
 {pubyear}{\edef\nowfnt@issuepubyear{#1}}
\define@key{nowfnt}
 {volume}{\def\nowfnt@issuevolume{#1}}

\define@key{nowfnt}
 {editedby}{\def\nowfnt@editedby{#1}}
\setkeys{nowfnt}{editedby=Edited by}

\define@key{nowfnt}
 {source}{\def\nowfnt@sourcetext{#1}}
\setkeys{nowfnt}{source=MISSING}

\newcommand\issuesetup[1]{\setkeys{nowfnt}{#1}}

%Default values
\setkeys{nowfnt}
 {
  doi=XXX,
  firstpage=1,
  isbn=XXX-X-XXXXX-XXX-X,
  eisbn=XXX-X-XXXXX-XXX-X,
  issue=XX,
  copyrightyear=\the\year,
  copyrightowner=\nowfnt@journalcopyrightowner,
  pubyear=\the\year,
  volume=XX,
  lastpage=\pageref{LastPage},
 }

\newcommand\printjournaltitleprefix[1][\ ]{%
 \ifdefempty{\nowfnt@journaltitleprefix}
 {}
 {\nowfnt@journaltitleprefix#1}}

\renewcommand\printjournaltitleprefix[1][]{} %deactivated 02.02.2016

\IfFileExists{microtype.sty}
 {\RequirePackage{microtype}}
 {\typeout{microtype not found}}

% only list chapters and sections in contents
\setcounter{tocdepth}{1}

% override some font size options
\renewcommand\normalsize{%
   \@setfontsize\normalsize\@xipt\@xivpt
   \abovedisplayskip 9\p@ \@plus2\p@ \@minus5\p@
   \abovedisplayshortskip \z@ \@plus3\p@
   \belowdisplayshortskip 5\p@ \@plus3\p@ \@minus3\p@
   \belowdisplayskip \abovedisplayskip
   \let\@listi\@listI
   }
\def\@xvpt{15}
\def\@xxviipt{27}
\renewcommand\scriptsize{\@setfontsize\scriptsize\@viipt\@viiipt}
\renewcommand\tiny{\@setfontsize\tiny\@vpt\@vipt}
\renewcommand\large{\@setfontsize\large\@xipt{13}}
\renewcommand\LARGE{\@setfontsize\LARGE\@xvpt{19}}
\renewcommand\Huge{\@setfontsize\Huge\@xxviipt{27}}

%\setlength\parskip{0pt}
% fonts
\newcommand\nowfnt@font@frontmattersection{\normalfont\bfseries\sffamily\LARGE}
\newcommand\nowfnt@font@frontmattersubsection{\normalfont\normalsize\sffamily\bfseries}
\newcommand\nowfnt@font@editorialboardsection{\normalfont\sffamily\bfseries\LARGE}
\newcommand\nowfnt@font@editorialboardsubsection{\normalfont\sffamily\normalsize\bfseries}
\newcommand\nowfnt@font@editorialboardvolumne{\mdseries}
\newcommand\nowfnt@font@editortext{\normalfont\footnotesize}
\newcommand\nowfnt@font@section{\normalfont\large\sffamily\bfseries\boldmath}
\newcommand\nowfnt@font@subsection{\normalfont\sffamily\large\bfseries\boldmath}
\newcommand\nowfnt@font@paragraph{\normalfont\sffamily\normalsize\bfseries\boldmath}
\newcommand\nowfnt@font@journalinfo{\normalfont\scriptsize\rmfamily}
\newcommand\nowfnt@font@partname{\normalfont\huge\bfseries \sffamily}
\newcommand\nowfnt@font@part{\normalfont\Huge\bfseries \sffamily}
\newcommand\nowfnt@font@chapternumber{\normalfont\sffamily\Huge\bfseries\centering}%
\newcommand\nowfnt@font@chaptertext{\normalfont\sffamily\LARGE\bfseries\centering\boldmath}%
\newcommand\nowfnt@font@header{\normalsize\normalfont\sffamily\slshape}
\newcommand\nowfnt@font@pagenumber{\normalsize\normalfont\sffamily}
\newcommand\nowfnt@font@title{\sffamily\bfseries\huge} %book and ebook
\newcommand\nowfnt@font@journaltitle{\LARGE \sffamily \bfseries} %journal
\newcommand\nowfnt@font@coverjournalname{\Large \sffamily \bfseries} %journal, coverpage
\newcommand\nowfnt@font@subtitle{\sffamily\fontsize{16pt}{21}\bfseries} %all
\newcommand\nowfnt@font@citation{\sffamily\footnotesize}

\newcommand\nowfnt@font@articledata  {\normalfont\footnotesize}
\newcommand\nowfnt@font@abstract        {\normalfont\normalsize}
\newcommand\nowfnt@font@authors      {\normalfont\normalsize}
\newcommand\nowfnt@font@affil        {\normalfont\normalsize\itshape}


\newcommand\nowfnt@contentsname{\contentsname}
% make table of contents sans serif
\renewcommand\tableofcontents{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\nowfnt@contentsname
        \@mkboth{%
           \nowfnt@font@header\nowfnt@contentsname}{\nowfnt@font@header\nowfnt@contentsname}}%
    {\sffamily \@starttoc{toc}}%
    \if@restonecol\twocolumn\fi
    }


\def\@makechapterhead#1{\vbox to 19pc{%
  \vspace*{15\p@}%
  {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
      \if@mainmatter
        {\nowfnt@font@chapternumber\thechapter\par}\addvspace{10pt}\nobreak
        \noindent\hrule\@width1\columnwidth\@height1\p@
        \vskip 14\p@
      \fi
    \fi
    \interlinepenalty\@M
    {\nowfnt@font@chaptertext#1\strut\par}\addvspace{10pt}\nobreak
    \noindent\hrule\@width1\columnwidth\@height1\p@
    \vfill%\vskip 14pc %12.11.2014 UF
  }}}
\def\@schapter#1{\if@twocolumn
                   \@topnewpage[\@makeschapterhead{#1}]%
                 \else
                   \@makeschapterhead{#1}%
                   \@afterheading
                 \fi}
\def\@makeschapterhead#1{%
\vbox to 16pc{%
  \vspace*{55\p@}%
  {\parindent \z@ \raggedright
    \interlinepenalty\@M
    {\nowfnt@font@chaptertext#1\strut\par}\addvspace{8pt}\nobreak
    \noindent\hrule\@width1\columnwidth\@height1\p@
  }\vfill}} %12.11.2014 UF vfill inserted

\ifbool{@nowfntplain}
{
 \renewcommand\chapter{\@startsection {chapter}{0}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {2.3ex \@plus.2ex}%
                                   {\normalfont\LARGE\bfseries}}
}

%% page headers
\RequirePackage{fancyhdr}

\providecommand\clearemptydoublepage{%
  \clearpage \if@twoside \ifodd\c@page \else
  \hbox{}\thispagestyle{empty}\newpage\if@twocolumn\hbox{}\newpage\fi\fi\fi}

\renewcommand{\headrulewidth}{0pt}%

% regular pages
\ifbool{@nowfntplain}
{\pagestyle{plain}}
{
\pagestyle{fancy}
  \fancyhead{}%
  \fancyheadoffset{0pt}%
  \fancyhead[RE]{\nowfnt@font@header\nouppercase \leftmark}%
  \fancyhead[LE,RO]{\nowfnt@font@pagenumber\thepage}%
  \fancyhead[LO]{\nowfnt@font@header\nouppercase \rightmark}%
  \fancyfoot[C]{}%

\fancypagestyle{plain}
 {%
   \fancyhead{}%
   \fancyfoot[C]{\nowfnt@font@pagenumber\thepage}%
 }

\fancypagestyle{abstractpage}{%
 \fancyhf{}%
 \lhead{%\nowfnt@font@header
        %\textit{\nowfnt@journaltitle, \nowfnt@issuepubyear, \nowfnt@issuevolume: \nowfnt@issuefirstpage--\nowfnt@issuelastpage}
        }
 \lfoot{\raisebox{\footskip}[0pt][0pt]{\usebox\nowfnt@article@databox}}}%06.11.2014

\fancypagestyle{titlepage}{%
 \fancyhf{}%
 \lhead{%\nowfnt@font@header
        %\textit{\nowfnt@journaltitle, \nowfnt@issuepubyear, \nowfnt@issuevolume: \nowfnt@issuefirstpage--\nowfnt@issuelastpage}
       }
 }%06.11.2014
}

\ifbool{@nowfntplain}
{}
{
\renewcommand{\chaptermark}[1]{\markboth{#1}{}}%

% make parts use \sffamily and empty pagestyle
% note: other section types redefined using titlesec
\def\@part[#1]#2{%
    \ifnum \c@secnumdepth >-2\relax
      \refstepcounter{part}%
      \addcontentsline{toc}{part}{\thepart\hspace{1em}#1}%
    \else
      \addcontentsline{toc}{part}{#1}%
    \fi
    \markboth{}{}%
    {\centering
     \thispagestyle{empty}%
     \interlinepenalty \@M
     \ifnum \c@secnumdepth >-2
       \nowfnt@font@partname \partname\nobreakspace\thepart
       \par
       \vskip 20\p@
     \fi
     \nowfnt@font@part #2\par}%
    \@endpart}
\def\@spart#1{%
    {\centering
     \interlinepenalty \@M
     \nowfnt@font@part #1\par}%
    \@endpart}
}
% section title fonts


\ifbool{@nowfntplain}
{}
{
\RequirePackage{titlesec}
\titleformat*{\section}{\nowfnt@font@section}
\titleformat*{\subsection}{\nowfnt@font@subsection}
\titleformat*{\subsubsection}{\nowfnt@font@paragraph}
\titleformat*{\paragraph}{\nowfnt@font@paragraph}
\titleformat*{\subparagraph}{\nowfnt@font@paragraph}
}

% small caption font
\RequirePackage[footnotesize,bf]{caption}

% appendices
\ifbool{@nowfntplain}
{\appto\appendix{\renewcommand\theHsection{app\thesection}}{}{}}
{
 \renewcommand\appendix
 {%
  \par
  \ifboolexpr{bool {@nowfntbook} or bool {@nowfntwrapperbook} or bool{@nowfntchaptertwoside}}
  {\cleardoublepage}{\clearpage}
  \thispagestyle{empty}%
  \addcontentsline{toc}{part}{Appendices}%
  \vspace*{\fill}%
  \begin{flushright}
   \huge\sffamily\bfseries Appendices
  \end{flushright}
  \vspace*{\fill}%
  \setcounter{chapter}{0}%
  \setcounter{section}{0}%
  \gdef\@chapapp{\appendixname}%
  \gdef\thechapter{\@Alph\c@chapter}%
 }
}
% abstract environment
\providecommand\abstractname{Abstract}
%\newenvironment{abstract}
% {%
%   \thispagestyle{empty}%
%   \section*{\abstractname}%
%   \ifboolexpr
%    {bool {@nowfntbook} or bool {@nowfntebook}}
%    {}
%    {\setcounter{page}{\nowfnt@issuefirstpage}}%
%   \sfcode`\.\@m
% }
% {%
%  \vfill\noindent
%  \rule{\textwidth}{0.5pt}\\[1pt]
%   {%
%    \footnotesize
%    \raggedright
%    \useissuedata{copyrightowner}. \emph{\nowfnt@articletitle}.
%    \printjournaltitleprefix
%    \usejournaldata{title},
%    vol.~\useissuedata{volume}, no.~\useissuedata{issue},
%    pp.~\useissuedata{firstpage}--\useissuedata{lastpage},
%    \useissuedata{pubyear}.\\
%    DOI: \useissuedata{doi}.\par
%   }%UF \par added
%  }

%%%%%%%%%%%%%%
%% new title command, copied from now-journal

% title need special treatment as it can be different in various locations (bib, header, toc, title)
\define@key{nowtitle}{bib}{\def\nowfnt@articlebibtitle{#1}}
\define@key{nowtitle}{header}{\def\nowfnt@articleheadertitle{#1}}
\define@key{nowtitle}{title}{\def\nowfnt@articletitle{#1}}
\define@key{nowtitle}{toc}{\def\nowfnt@articletoctitle{#1}}

 \newcommand\nowfnt@articletitle{}
 \newcommand\nowfnt@articlebibtitle{}
 \newcommand\nowfnt@articleheadertitle{}
 \newcommand\nowfnt@articletoctitle{}

\ifbool{@nowfntplain}
{\renewcommand\title[2][]{\gdef\@title{#2}}}
{
\renewcommand\title[2][]{%
 \setkeys{nowtitle}{bib={#2},header={#2},title={#2},toc={#2}}
 \setkeys{nowtitle}{#1}}
}
\def\nowfnt@titlefootnote{}
\newcommand\titlefootnote[1]{\def\nowfnt@titlefootnote{#1}}

%% new autor command copied from now-journal.cs

%authors need very special treatment.
%There three places where author names (lists) are used: title (fullname), header (last), bib-file
%authors can have three name parts: First, Junior, Last (in bib-Order).

\def\nowfnt@splitauthor#1,#2,#3,#4\@nil{%
 \def\nowfnt@tempa{!}%
 \def\nowfnt@firstname{}\def\nowfnt@juniorpart{}\def\nowfnt@lastname{}%
 \def\nowfnt@tempfirstarg{#1}\def\nowfnt@tempsecondarg{#2}\def\nowfnt@tempthirdarg{#3}%
 \ifx\nowfnt@tempthirdarg\nowfnt@tempa
   \ifx\nowfnt@tempsecondarg\nowfnt@tempa
    \typeout{First name missing}%
    \def\nowfnt@lastname{#1}%
   \else
    \def\nowfnt@lastname{#1}%
    \def\nowfnt@firstname{#2}%
   \fi
 \else
   \def\nowfnt@lastname{#1}%
   \def\nowfnt@juniorpart{#2}%
   \def\nowfnt@firstname{#3}%
 \fi%
   \trim@spaces@in\nowfnt@lastname
   \trim@spaces@in\nowfnt@juniorpart
   \trim@spaces@in\nowfnt@firstname}

\newcounter{nowfnt@author@cnt}
\newcounter{nowfnt@affil@cnt}
\newcounter{nowfnt@affilmax@cnt}
\setcounter{nowfnt@affil@cnt}{0}

\newcommand\nowfnt@authorlist@bibsep{}
\newcommand\nowfnt@authorlist@bib{}

\renewcommand\author[2][\the\numexpr\c@nowfnt@affil@cnt+1]{%
 \stepcounter{nowfnt@author@cnt}%
 \nowfnt@splitauthor#2,!,!,\@nil%
 \edef\next{\noexpand\g@addto@macro\noexpand\nowfnt@authorlist@bib{%
        \nowfnt@authorlist@bibsep\unexpanded{#2}}}\next
 \def\nowfnt@authorlist@bibsep{ and }%
 \expandafter\edef\csname nowfnt@authorlastname@\the\c@nowfnt@author@cnt\endcsname{\unexpanded\expandafter{\nowfnt@lastname}}%
 \expandafter\edef\csname nowfnt@authorfullname@\the\c@nowfnt@author@cnt\endcsname%
           {\unexpanded\expandafter{\nowfnt@firstname}\noexpand~\unexpanded\expandafter{\nowfnt@lastname}%
             \ifx\nowfnt@juniorpart\@empty
             \else,\noexpand~\unexpanded\expandafter{\nowfnt@juniorpart}%
             \fi
            }%
 \expandafter\edef\csname nowfnt@authoraffilmark@\the\c@nowfnt@author@cnt\endcsname{#1}%
 \nowfnt@store@authorlist@header
 \nowfnt@store@authorlist@toc
}

\newcommand\affil[2][\the\c@nowfnt@affil@cnt]{%
 \stepcounter{nowfnt@affil@cnt}%
 \c@nowfnt@affilmax@cnt=\numexpr\c@nowfnt@author@cnt\relax
 \expandafter\def\csname nowfnt@affil@\the\c@nowfnt@affil@cnt\endcsname{#2}%
 \expandafter\edef\csname nowfnt@affilmark@\the\c@nowfnt@affil@cnt\endcsname{#1}%
 }

%Commands to create authorlists
% 1. a list for the header: Lastnames, if > 2 authors: et al,
\newcommand\nowfnt@authorlist@header{}%
\newcommand\nowfnt@store@authorlist@header{%
 \ifcase\c@nowfnt@author@cnt
 \or
 \edef\nowfnt@authorlist@header{\unexpanded\expandafter\expandafter\expandafter{\csname nowfnt@authorlastname@1\endcsname}}%
 \or
 \edef\nowfnt@authorlist@header{\unexpanded\expandafter\expandafter\expandafter{\csname nowfnt@authorlastname@1\endcsname} and
                             \unexpanded\expandafter\expandafter\expandafter{\csname nowfnt@authorlastname@2\endcsname}}%
 \else
 \edef\nowfnt@authorlist@header{\unexpanded\expandafter\expandafter\expandafter{\csname nowfnt@authorlastname@1\endcsname} et al.}%
 \fi}%

% 1. a list for the wrapper  toc.
\newcommand*\nowfnt@authorlist@toc{}

\newcommand\nowfnt@store@authorlist@toc{%
  \edef\nowfnt@authorlist@toc{\unexpanded\expandafter\expandafter\expandafter{\csname nowfnt@authorfullname@1\endcsname}}%
  \ifnum\c@nowfnt@author@cnt=1\relax
  \else
  \c@nowfnt@tempcnt=2\relax
    \whiledo{\c@nowfnt@tempcnt<\c@nowfnt@author@cnt}
      {\edef\nowfnt@authorlist@toc%
         {\unexpanded\expandafter{\nowfnt@authorlist@toc,}
          \unexpanded\expandafter\expandafter\expandafter{\csname nowfnt@authorfullname@\the\c@nowfnt@tempcnt\endcsname}}%
       \stepcounter{nowfnt@tempcnt}%
      }%
      \edef\nowfnt@authorlist@toc%
       {\unexpanded\expandafter{\nowfnt@authorlist@toc} and
        \unexpanded\expandafter\expandafter\expandafter{\csname nowfnt@authorfullname@\the\c@nowfnt@author@cnt\endcsname}}%
  \fi
}


\define@key{nowauthorlist}{bib}{\def\nowfnt@authorlist@bib{#1}}
\define@key{nowauthorlist}{header}{\def\nowfnt@authorlist@header{#1}}
\define@key{nowauthorlist}{toc}{\def\nowfnt@authorlist@toc{#1}}
\newcommand\authorlists[1]{\setkeys{nowauthorlist}{#1}}

%%%%% new abstract environment / copied from now-journal.cls

\newcommand\nowfnt@abtractendhook{}

\ifbool{@nowfntplain}
{}
{
 \newenvironment{abstract}
 {%
  \ifboolexpr
   {bool {@nowfntbook} or bool {@nowfntebook}}
   {}
   {\setcounter{page}{\nowfnt@issuefirstpage}}%
  \ifboolexpr{bool {@nowfntchapter}}
    {%
     \ifcsname nowfntchapterfirstpage\endcsname
     \setcounter{page}{\nowfntchapterfirstpage}%
     \fi
    }{}%
  \nowfnt@font@abstract\clubpenalty=10000%
  \begin{list}{}{\rightmargin\leftmargin}%
          \item\relax
 \hspace*{-\leftmargin}\noindent\makebox[\linewidth][l]{\rule{\textwidth}{0.5pt}}\par\vspace{\medskipamount}\nobreak
          ABSTRACT\par\@afterheading}
 {\par\nobreak\hspace*{-\leftmargin}\makebox[\linewidth][l]{\rule{\textwidth}{0.5pt}}\end{list}
  \addvspace{\medskipamount}\nowfnt@abtractendhook}
 }

\newcommand\nowfnt@printaffilmark[1]{%#1=author number
  \textsuperscript{\csname nowfnt@authoraffilmark@#1\endcsname}%
 }




\DeclareRobustCommand\nowfnt@printauthors{%
 \ifnum\c@nowfnt@author@cnt=0\relax
 \else
   \csname nowfnt@authorfullname@1\endcsname
      \nowfnt@printaffilmark{1}%
   \ifnum\c@nowfnt@author@cnt=1\relax
   \else
   \c@nowfnt@tempcnt=2\relax
    \whiledo{\c@nowfnt@tempcnt<\c@nowfnt@author@cnt}
      {,\space\csname nowfnt@authorfullname@\the\c@nowfnt@tempcnt\endcsname
       \nowfnt@printaffilmark{\the\c@nowfnt@tempcnt}%
       \stepcounter{nowfnt@tempcnt}%
      }
    and \csname nowfnt@authorfullname@\the\c@nowfnt@author@cnt\endcsname
        \nowfnt@printaffilmark{\the\c@nowfnt@author@cnt}%
   \fi
   \ifcsname nowfnt@creditlinetext\endcsname
     \footnote{\nowfnt@creditlinetext}%
   \fi
\fi}



\newcommand\nowfnt@printaffils{%
 \begingroup\nowfnt@font@affil
  \ifnum\c@nowfnt@affil@cnt > 0
  \c@nowfnt@tempcnt=0\relax
    \whiledo{\c@nowfnt@tempcnt<\c@nowfnt@affil@cnt}
      {\par
       \stepcounter{nowfnt@tempcnt}%
        \textsuperscript{\upshape\csname nowfnt@affilmark@\the\c@nowfnt@tempcnt\endcsname}%
        \csname nowfnt@affil@\the\c@nowfnt@tempcnt\endcsname
      }%
  \fi
 \endgroup
  }


\newcommand\creditline[2][*]{%
 \def\nowfnt@creditlinemark{#1}%
 \def\nowfnt@creditlinetext{#2}}

\newsavebox\nowfnt@article@databox{}
\sbox\nowfnt@article@databox{}
\newcommand\nowfnt@article@databox@wd{\dimexpr\textwidth\relax}
\newcommand\articledatabox[1]{%
 \AtBeginDocument{%
 \sbox\nowfnt@article@databox{\parbox[b]{\nowfnt@article@databox@wd}
  {\nowfnt@font@articledata\rule{\nowfnt@article@databox@wd}{0.5pt}\par #1}}}}

\newcommand\nowfnt@defaulttextcontent{XXXXXXXX}

\newcommand\articleDOI[1]{\def\nowfnt@articleDOI{#1}}

%\renewcommand\articleDOI[1]{%
% \nowfnt@split@DOI#1//\@nil}

\newcommand\nowfnt@split@DOI{}
\def\nowfnt@split@DOI#1/#2/#3\@nil{%call as \nowfnt@split@DOI<arg>//\@nil
 \def\@tempa{#3}%
 \ifx\@tempa\@empty %kein slash im Argument
   \ifx\nowfnt@journaldoiprefix\@empty
    \ClassWarning{now-journal}{Journal DOI prefix missing!}%
    \def\nowfnt@articleDOI{XX.XXX/#1}%
   \else
    \def\nowfnt@articleDOI{\nowfnt@journaldoiprefix/#1}%
   \fi
 \else
  \def\nowfnt@articleDOI{#1/#2}%
 \fi}


\let\nowfnt@articleDOI\nowfnt@defaulttextcontent

\newcommand\articlecopyrightholder[1]{\def\nowfnt@articlecopyrightholder{#1}}
\let\nowfnt@articlecopyrightholder\nowfnt@defaulttextcontent

\AtBeginDocument{%
\sbox\nowfnt@article@databox{\parbox[b]{\nowfnt@article@databox@wd}
  {\nowfnt@font@articledata\rule{\nowfnt@article@databox@wd}{0.5pt}\par
   ISSN \ifx\nowfnt@journalissnprint\@empty \nowfnt@journalissnonline\else\nowfnt@journalissnprint\fi; DOI \nowfnt@articleDOI\\
   \copyright\nowfnt@issuecopyrightyear~\nowfnt@articlecopyrightholder}}}


\newcommand\disclaimer[1]{\def\nowfnt@disclaimer{#1}}

%%%%%%%%%%%% End of new abstract code


% acknowledgments environment ????
\newcommand\acknowledgmentsname{Acknowledgments}
\newenvironment{acknowledgments}
  {%
   \chapter*{\acknowledgmentsname}%
   \@mkboth{\acknowledgmentsname}{\acknowledgmentsname}%
   \addcontentsline{toc}{chapter}{\acknowledgmentsname}
   \sfcode`\.\@m
  }
  {}

\newcommand\acknowledgementsname{Acknowledgements}
\newenvironment{acknowledgements}
  {%
   \chapter*{\acknowledgementsname}%
   \@mkboth{\acknowledgementsname}{\acknowledgementsname}%
   \addcontentsline{toc}{chapter}{\acknowledgementsname}
   \sfcode`\.\@m
  }
  {}

% don't include headers on empty pages
\RequirePackage{emptypage}




% front matter
\newcommand\nowfnt@maintitleauthorlist{}

\ifbool{@nowfntplain}
{\newcommand\maintitleauthorlist[1]{\gdef\@author{#1}}}
{
\newcommand\maintitleauthorlist[1]{%
 \def\nowfnt@maintitleauthorlist{#1}}
}


\newcommand\nowfnt@printauthorsbooktitle{%
 \begingroup
 \fontsize{13}{15}\sffamily
 \def\and
  {%
   \end{tabular}%
   \par\vspace*{1.5ex}
   \begin{tabular}[t]{@{}r@{}}\bfseries
  }%
 \begin{tabular}[t]{@{}r@{}}
 \bfseries\nowfnt@maintitleauthorlist
 \end{tabular}%
 \endgroup
 }

\newsavebox\nowfnt@tmpabox

\newcommand\nowfnt@printauthorsbooktitletwocolumn{%
 \begingroup
 \setcounter{nowfnt@tempcnt}{0}
 \fontsize{13}{15}\sffamily
 \def\and
  {%
   \end{tabular}%
   \stepcounter{nowfnt@tempcnt}%
   \ifodd \c@nowfnt@tempcnt
     \expandafter\@firstoftwo
   \else
     \expandafter\@secondoftwo
   \fi
   {&}
   {\tabularnewline\noalign{\vspace{1.5ex}}}%\par\vspace{1.5ex}%
   \begin{tabular}[t]{@{}r@{}}\bfseries
  }%
 \begin{tabular}{@{}rr@{}}
 \begin{tabular}[t]{@{}r@{}}
 \bfseries\nowfnt@maintitleauthorlist
 \end{tabular} %
 \end{tabular}
 \par
 \endgroup
 }



\newcolumntype\journalauthorcoltype{@{}>{\centering\arraybackslash}p{0.48\textwidth}@{}}

\newcommand\nowfnt@printauthorsjournaltitle{%
 \begingroup
 \fontsize{13}{15}\sffamily
 \def\and
  {%
   \end{tabular}%
   \hfill
   \begin{tabular}[t]{\journalauthorcoltype}
  }%
 \begin{tabular}[t]{\journalauthorcoltype}
  \@author
 \end{tabular}%
 \endgroup
 }
%% title page

%% subtitle command:
\newcommand\subtitle[1]{\gdef\now@subtitle{#1}}
\newcommand\now@subtitle{}
\gdef\now@subtitle{}

\newcommand\nowfnt@booktitlepage
 {
  \begin{titlepage}
  \parindent \z@
  % title page
  %\thispagestyle{empty}
  \vspace*{\fill}
  \begin{flushright}
   \begin{minipage}{0.8\textwidth}
    \begin{flushright}
    \renewcommand{\baselinestretch}{1.1}
    \nowfnt@font@title
    \nowfnt@articletitle
    \ifdefempty{\now@subtitle}
     {}
     {\par\bigskip\nowfnt@font@subtitle\now@subtitle}%
    \end{flushright}
   \end{minipage}
  \end{flushright}
  \vspace*{\fill}
  \end{titlepage}
 }

\newbool{authortwocolumn}
\newcommand\nowfnt@titlepagestyle{empty}
\newcommand\nowfnt@bookauthortitlepage
 {
  \newpage
  \thispagestyle{\nowfnt@titlepagestyle}
  %\begin{flushright}
  \begingroup
  \raggedleft
  {%
   \renewcommand{\baselinestretch}{1.05}% statt 1.2 18.05.2016
   \nowfnt@font@title
   \if@nowfntjournal
     \textcolor{\nowfnt@journalcolor}{\nowfnt@font@coverjournalname\nowfnt@journaltitle}\par
   \fi
   \nowfnt@articletitle
    \ifdefempty{\now@subtitle}
     {}
     {\par\bigskip\nowfnt@font@subtitle\now@subtitle}%
    \par
  }
  \vspace{2.5ex}
  \noindent\rule{\textwidth}{2pt}
  \par\vspace{1ex}
  \if@nowfntjournal
   \parbox{\linewidth}{\sffamily\footnotesize\textbf{Suggested Citation: }\nowfntstandardcitation\par}\par\bigskip
  \fi
  \if@nowfntwrapper
  \begingroup
  \fontsize{13}{15}\sffamily\bfseries
   \nowfnt@editedby
  \par\vspace{1ex}
  \endgroup
  \fi
  \ifbool{authortwocolumn}
  {\nowfnt@printauthorsbooktitletwocolumn}
  {\nowfnt@printauthorsbooktitle}%
  \par\vfill
  \includegraphics[width=53bp]{now_logo}\\
  \includegraphics[width=81bp]{essence_logo}\\
  \if@nowfntjournal
  \raisebox{\dp\strutbox}[0pt][0pt]{\fbox{\parbox[b]{\dimexpr\linewidth-130bp}{\scriptsize\sffamily\nowfnt@prohibitiontext}}}
  \fi\hfill
  Boston --- Delft
  \par\endgroup %\end{flushright}
 }

\newcommand\nowfnt@bookpublisherpage
 {
  \newpage
  \thispagestyle{empty}
  \begingroup
  \parindent 0pt
  \begin{minipage}{\textwidth}
  \bfseries\sffamily\Large
  \printjournaltitleprefix \nowfnt@journaltitle
  \end{minipage}
  \par
  \parskip6pt
  \footnotesize
  \par\vfill
  \nowfnt@publicationaddress
  \par
  \ifbool{@nowfntwrapper}{\csname nowfnt@sourceintro\endcsname\csname nowfnt@sourcetext\endcsname\csname nowfnt@sourcepunct\endcsname}{\nowfnt@howtocitetext}
  \par
  \nowfnt@printedoninfotext
  \par
  \ifbool{@nowfntwrapper}{\nowfnt@NOWcopyrightlinetext}{\nowfnt@copyrightlinetext}
  \par
  \vfill
  {\scriptsize\nowfnt@legalrightstext\par}
  \endgroup
 }




% editorial board
\newcommand\nowfnt@bookeditorboardpage
 {%
  \newpage
  \input{\nowfnt@journalshorttitle-editorialboard}
 }

\newcommand\nowfnt@bookeditorialscopepage
 {%% editorial scope
  \newpage % page vi -- blank
  \thispagestyle{empty}
  \begingroup %same definition as for board
  \renewcommand\section{\@startsection {section}{1}{\z@}%
                       {-3.5ex \@plus -1ex \@minus -.2ex}%
                       {2ex \@plus.2ex}%
                       {\nowfnt@font@editorialboardsection\centering}}%
  \section*{Editorial Scope}
  \subsection*{Topics}
  \small\raggedright
  \usejournaldata{aimsandscope}
  \par
  \endgroup
  \subsection*{Information for Librarians}
  \usejournaldata{libraryinfo}
}

\newcommand\nowfnt@journaltitlepage
{
 \begin{titlepage}
  \thispagestyle{empty}
  \hbox to \textwidth{%
   \vbox{\raggedright
    \hsize=.6\textwidth \nowfnt@font@journalinfo
   {\hangindent=3.5ex
    \hangafter=1
    \printjournaltitleprefix \usejournaldata{title}\par}%
   Vol.~\useissuedata{volume}, No.~\useissuedata{issue} (\useissuedata{pubyear})
  \useissuedata{firstpage}--\useissuedata{lastpage}\\
  {\hangindent=3.5ex
   \hangafter=1
   \copyright\ \useissuedata{copyrightyear} \useissuedata{copyrightowner}\par}%
   DOI: \useissuedata{doi}}\hss
  \vbox{\hsize=.4\textwidth
    \raggedleft
    \includegraphics[width=53bp]{now_logo}\\
    \includegraphics[width=81bp]{essence_logo}\\
    \par
   }}\par
  \let\footnoterule\relax
  \let\footnote\thanks
  \null\vfil
  \vskip 60\p@
  \begin{center}%
  {\nowfnt@font@journaltitle \nowfnt@articletitle
    \ifdefempty{\now@subtitle}
     {}
     {\par\bigskip\nowfnt@font@subtitle\now@subtitle}%
    \par
  }%
  \vskip 3em%
  {%
   \large
   \lineskip .75em
   \nowfnt@printauthorsjournaltitle\par
  }%
   \end{center}\par
   \@thanks
   \vfill
   \end{titlepage}
  }

\ifbool{@nowfntplain}
{}
{
\renewcommand\maketitle{%
 \if@nowfntbook % -- for 'book' version
  \nowfnt@booktitlepage
  \nowfnt@seriespage
  \clearemptydoublepage
  \nowfnt@bookauthortitlepage
  \nowfnt@bookpublisherpage
  \nowfnt@bookeditorboardpage
  \nowfnt@bookeditorialscopepage
 \fi
 \if@nowfntjournal
   %\nowfnt@journaltitlepage
   \renewcommand\nowfnt@titlepagestyle{titlepage}%
   \nowfnt@bookauthortitlepage
 \fi
 \if@nowfntsample
  %\nowfnt@journaltitlepage
 \nowfnt@bookauthortitlepage
 \fi
 \if@nowfntebook %now same a book
  \let\nowfnt@issueisbn\nowfnt@issueeisbn
  \nowfnt@booktitlepage
  \nowfnt@seriespage
  \cleardoublepage
  \nowfnt@bookauthortitlepage
  \nowfnt@bookpublisherpage
  \nowfnt@bookeditorboardpage
  \nowfnt@bookeditorialscopepage%
%  \nowfnt@bookauthortitlepage
%  \nowfnt@seriespage
%  \nowfnt@bookpublisherpage
%  \nowfnt@bookeditorboardpage
%  \nowfnt@bookeditorialscopepage
 \fi
 \if@nowfntwrapper % -- for wrapper 'book' version
  \nowfnt@booktitlepage
  \nowfnt@seriespage
  \ifboolexpr{bool {@nowfntwrapperbook}}{\clearemptydoublepage}{\clearpage}
  \renewcommand\nowfnt@titlepagestyle{empty}
  \nowfnt@bookauthortitlepage
  \nowfnt@bookpublisherpage
  \nowfnt@bookeditorboardpage
  \nowfnt@bookeditorialscopepage
 \fi
 \setcounter{footnote}{0}
}
}

\ifbool{@nowfntplain}
{\newcommand\makeabstracttitle{}}
{
\newcommand\makeabstracttitle{%
\begingroup
 \setfnsymbol{wiley}%
 \renewcommand\thefootnote{\fnsymbol{footnote}}%
 \begingroup
 \raggedright
 \thispagestyle{abstractpage}%
   \ifboolexpr
   {bool {@nowfntbook} or bool {@nowfntebook}}%
   {}
   {\setcounter{page}{\nowfnt@issuefirstpage}}%
  \ifboolexpr{bool {@nowfntchapter}}
    {%
     \ifcsname nowfntchapterfirstpage\endcsname
     \setcounter{page}{\nowfntchapterfirstpage}%
     \fi
    }{}%
 \enlargethispage{\dimexpr-\ht\nowfnt@article@databox-\footskip+\baselineskip}%06.11.2014
 \nowfnt@font@title{%
  \def\footnote{\ClassError{nowfnt.cls}{Don't use \string\footnote\space in the title!^^J
                Use \string\titlefootnote\space after \string\title\space instead}{}{}}%
     \nowfnt@articletitle}%
 \ifx\nowfnt@titlefootnote\@empty\else\footnote{\nowfnt@titlefootnote}\fi\par\medskip
 \nowfnt@font@authors\nowfnt@printauthors\par\medskip
 \nowfnt@printaffils\par
 \endgroup
 \ifcsname nowfnt@disclaimer\endcsname\medskip\begingroup\noindent\small\textbf{Disclaimer:~}\ignorespaces\nowfnt@disclaimer\par\endgroup\fi
 \endgroup\setcounter{footnote}{0}}
}
\newcommand\journalchiefeditor[2]
 {%
  \begin{tabular}{@{}l}
   \bfseries  #1\\ #2
  \end{tabular}%
 }

\newcommand\journaleditor{}


\newcommand\nowfnt@seriespage{%
\IfFileExists{\jobname-seriespage.tex}
 {\newpage\nowfnt@seriespagetitle
  \input{\jobname-seriespage}%
 }
 {%
  \IfFileExists{\nowfnt@journalshorttitle-seriespage.tex}
  {\newpage\nowfnt@seriespagetitle
   \input{\nowfnt@journalshorttitle-seriespage}%
  }
  {%
   \IfFileExists{seriespage.tex}
   {\newpage\nowfnt@seriespagetitle
    \input{seriespage}%
   }
   {
   \ClassWarning{nowfnt}{No series page found}{}
   }%
  }%
 }%
}

\newenvironment{editorialboard}
 {\parindent0pt\relax%
  \renewcommand\section{\@startsection {section}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {2ex \@plus.2ex}%
                                   {\nowfnt@font@editorialboardsection\centering}}%
  \renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-3.5ex\@plus -1ex \@minus -2.2ex}%
                                     {1ex \@plus .2ex}%
                                     {\nowfnt@font@editorialboardsubsection}}%
 \c@secnumdepth=0 %
 \parskip=5pt minus 3pt
 \def\journaleditor##1##2{\par\begin{tabular}[t]{@{}>{\raggedright}p{\linewidth}@{}}##1\\\emph{##2}\end{tabular}}%
 \thispagestyle{empty}%
 \begin{samepage}
 \section*{%
  \printjournaltitleprefix \usejournaldata{title}\\
  \strut{\nowfnt@font@editorialboardvolumne Volume \useissuedata{volume}, Issue \useissuedata{issue}, \useissuedata{pubyear}}\\[0.15ex]
  Editorial Board}%
 \nowfnt@font@editortext}
 {\par\end{samepage}}

% don't leave empty page before mainmatter for journal version
\ifbool{@nowfntplain}
 {
  \providecommand\mainmatter{} %for plain version
  \providecommand\backmatter{} %for plain version
 }
 {
  \renewcommand\mainmatter
  {%
  \ifboolexpr{bool {@nowfntbook} or bool {@nowfntwrapperbook}}{\clearemptydoublepage}{\clearpage}%
  \pagestyle{fancy}
  \@mainmattertrue
  %\hypersetup{pageanchor=true}%
  \pagenumbering{arabic}%
  }
 }

%no page numbers in frontmatter
\ifbool{@nowfntplain}
{
 \providecommand\frontmatter{} %for plain version
}
{
\renewcommand\frontmatter{%
    \cleardoublepage
  \pagestyle{empty}%
  \@mainmatterfalse
  %\hypersetup{pageanchor=false}%
  \pagenumbering{roman}}
}
% some default packages worth including
\RequirePackage{url}
\RequirePackage{graphicx}
\RequirePackage{lmodern}
\RequirePackage[T1]{fontenc}
\RequirePackage{microtype}

% bibliography style
\providecommand{\bibname}{References}% %for plain style
\if@nowfntbiber
  \expandafter\@firstoftwo
 \else
  \expandafter\@secondoftwo
 \fi
   {\RequirePackage{nowfnt-biblatex}}
   {%
    \ifdefstring\nowfnt@journalbibstyle{numeric}
    {
     \RequirePackage[numbers]{natbib}
     \bibliographystyle{nowfntsort}
    }
    {
     \RequirePackage{natbib}
     \bibliographystyle{nowfntsort}
    }
    \def\now@addbib@resource{}%
    \def\now@bibcomma{}
    \def\addbibresource#1
     {%
      \filename@parse{#1}%
      \ifx\filename@ext\relax
       \PackageWarning{now-journal}{You forgot the extension "bib"}{}
      \else
       \edef\now@tempa{\now@bibcomma\filename@area\filename@base}%
       \gdef\now@bibcomma{,}%
       \expandafter\g@addto@macro\expandafter\now@addbib@resource\expandafter{\now@tempa}%
      \fi
     }
    \newcommand\printbibliography[1][]
     {%
      \ifx\now@addbib@resource\@empty
       %\PackageError{nowfnt.cls}{No bib-file declared}{}
      \else
       \renewcommand{\bibname}{References}%
       \expandafter\bibliography\expandafter{\now@addbib@resource}%
      \fi
     }%
    \renewcommand{\bibfont}{\small\interlinepenalty 10000\relax}
    \setlength{\bibsep}{1ex}
    \renewcommand\bibsection
     {%
      \chapter*{\bibname}
      \markboth{\bibname}{\bibname}%
      \addcontentsline{toc}{chapter}{\bibname}
     }
    }

\renewcommand{\bibname}{References}

% theorem environments
\RequirePackage{amsthm}

% custom length changes below needed to ensure that lists
% contained in theorem and proof environments are properly
% indented as they should be; misalignment presumably due
% to some weird package conflict

\newtheoremstyle{now}
{}
{}
{}
{}
{\sffamily\bfseries}
{.}
{.5em}
{}


\theoremstyle{now}
\ifbool{@nowfntplain}
{\newtheorem{theorem}{Theorem}[section]
 \newtheorem{lemma}[theorem]{Lemma}
 \newtheorem{corollary}[theorem]{Corollary}
 \newtheorem{proposition}{Proposition}[section]
 \newtheorem{remark}{Remark}[section]
 \newtheorem{definition}{Definition}[section]
 \newtheorem{example}{Example}[section]
 }
{
 \newtheorem{theorem}{Theorem}[chapter]
 \newtheorem{lemma}[theorem]{Lemma}
 \newtheorem{corollary}[theorem]{Corollary}
 \newtheorem{proposition}{Proposition}[chapter]
 \newtheorem{remark}{Remark}[chapter]
 \newtheorem{definition}{Definition}[chapter]
 \newtheorem{example}{Example}[chapter]
}

% high-level typesetting style
%\sloppy
\flushbottom
\frenchspacing

% page geometry
\RequirePackage[vcentering,hcentering]{geometry}
\ifbool{@nowfntplain}
{}
{\geometry{papersize={156mm,234mm},total={118mm,180mm},headheight=14.4pt}}

\normalsize
\geometry{heightrounded}


\ifcsname tikzexternalrealjob\endcsname \else
\RequirePackage[%
 colorlinks    = true,
 allcolors     = blue,
 plainpages    = false,
 pdfpagelabels = true,
  ]{hyperref}
\RequirePackage{bookmark}
\fi
%% setting up discard pages




\newbool{nowfnt@disardpages}
\boolfalse{nowfnt@disardpages}

\newcommand\nowfnt@discardpagesinsample{%
 \ifboolexpr{bool{@nowfntsample} and bool{nowfnt@disardpages}}
   {\AtBeginShipoutDiscard}{}}

\ifbool{@nowfntplain}
{\newcommand\stopsamplehere{}}
{\newcommand\stopsamplehere{%
 \clearpage
 \booltrue{nowfnt@disardpages}}
}

\ifbool{@nowfntsample}
 {\appto\backmatter{\boolfalse{nowfnt@disardpages}}
  \AtBeginShipout{\nowfnt@discardpagesinsample}}
 {}

%wrapper code
\ifcsname tikzexternalrealjob\endcsname \else
\if@nowfntjournal
  \IfFileExists{\jobname-nowbook.tex}{}
  {%
   \immediate\openout\@unused=\jobname-nowbook.tex
   \immediate\write\@unused{\unexpanded{\PassOptionsToClass{book,biber}{nowfnt}}}
   \immediate\write\@unused{\noexpand\input{\jobname}}
   \immediate\closeout\@unused
  }
  \IfFileExists{\jobname-nowebook.tex}{}
  {%
   \immediate\openout\@unused=\jobname-nowebook.tex
   \immediate\write\@unused{\unexpanded{\PassOptionsToClass{ebook,biber}{nowfnt}}}
   \immediate\write\@unused{\noexpand\input{\jobname}}
   \immediate\closeout\@unused
  }
 \IfFileExists{\jobname-nowsample.tex}{}
  {%
   \immediate\openout\@unused=\jobname-nowsample.tex
   \immediate\write\@unused{\unexpanded{\PassOptionsToClass{sample,biber}{nowfnt}}}
   \immediate\write\@unused{\noexpand\input{\jobname}}
   \immediate\closeout\@unused
  }
 \IfFileExists{\jobname-nowchapter.tex}{}
  {%
   \immediate\openout\@unused=\jobname-nowchapter.tex
   %\immediate\write\@unused{\unexpanded{\newcommand\nowfntchapterfirstpage{1}}\@percentchar adapt to first page number}
   \immediate\write\@unused{\unexpanded{\PassOptionsToClass{chapter,biber}{nowfnt}}}
   \immediate\write\@unused{\noexpand\input{\jobname}}
   \immediate\closeout\@unused
  }
 \IfFileExists{\jobname-nowechapter.tex}{}
  {%
   \immediate\openout\@unused=\jobname-nowechapter.tex
   %\immediate\write\@unused{\unexpanded{\newcommand\nowfntchapterfirstpage{1}}\@percentchar adapt to first page number}
   \immediate\write\@unused{\unexpanded{\PassOptionsToClass{echapter,biber}{nowfnt}}}
   \immediate\write\@unused{\noexpand\input{\jobname}}
   \immediate\closeout\@unused
  }
 \IfFileExists{\jobname-nowplain.tex}{}
  {%
   \immediate\openout\@unused=\jobname-nowplain.tex
   \immediate\write\@unused{\unexpanded{\PassOptionsToClass{plain,biber}{nowfnt}}}
   \immediate\write\@unused{\noexpand\input{\jobname}}
   \immediate\closeout\@unused
  }
\fi
\fi

%% preliminiray \nowinclude code

\newcommand\nowfnt@wrappertocentry{XXXXXXXXXXXXXXX \\ XXXXXXXXXXXXXXXXXX toc entry missing}
\newcommand\nowfnt@wrappertocpages {\thepage}
\newcommand\nowfnt@wrapperfirstpage{1}

\define@key{nowfntwrapper}
 {toc}{\def\nowfnt@wrappertocentry{#1}}

\define@key{nowfntwrapper}
 {author}{\def\nowfnt@wrappertocauthors{#1}}

\define@key{nowfntwrapper}
 {authors}{\def\nowfnt@wrappertocauthors{#1}}

\define@key{nowfntwrapper}
 {pages}{\def\nowfnt@wrappertocpages{#1}}

\define@key{nowfntwrapper}
 {firstpage}{\def\nowfnt@wrapperfirstpage{#1}}

\newcounter{nowfnt@nowinclude}

\newcommand\nowinclude[2][]{%
 \if@nowfntwrapperbook
  \clearemptydoublepage
 \fi
 \stepcounter{nowfnt@nowinclude}%
 \ifnum \c@nowfnt@nowinclude=1\relax \pagenumbering{arabic}\fi
 \setkeys{nowfntwrapper}{#1}%
 \refstepcounter{chapter}%
 \addtocontents{toc}{%
      \protect\contentsline{chapter}{\nowfnt@wrappertocentry}{\nowfnt@wrappertocpages}{\@currentHref}}%
 \addtocontents{toc}{%
  \nowfnt@wrapperprinttocauthors{\nowfnt@wrappertocauthors}%
  }
 %\addcontentsline{toc}{chapter}{\nowfnt@wrappertocentry}{}%
 \includepdf[pages=\nowfnt@wrapperfirstpage-]{#2}%
 }

\DeclareRobustCommand\nowfnt@wrapperprinttocauthors[1]{%
 \parbox[t]{\dimexpr\linewidth-\@pnumwidth}{\itshape #1}\par}

\if@nowfntwrapper
 \renewcommand*\l@chapter[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 1.0em \@plus\p@
    \setlength\@tempdima{0pt}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}
\fi

\if@nowfntchapter
\@ifundefined{nowpredecessor}
 {\setcounter{page}{1}\newcommand\nowfntchapterfirstpage{1}}
 {\externaldocument[page-]{\nowpredecessor}%
  \setcounter{page}{\getpagerefnumber{page-LastPage}}%
  \stepcounter{page}%
  %
  \if@nowfntchaptertwoside
   \ifodd\c@page\else\stepcounter{page}\fi
  \fi
  \edef\nowfntchapterfirstpage{\number\c@page}
  %\fi
  }
\fi

%mostly copied from now-journal.cls
\newcounter{nowfnt@article@cnt}
\newcommand\addarticle[2][]{%
 \stepcounter{nowfnt@article@cnt}%
 \expandafter\def\csname nowfnt@article@\the\c@nowfnt@article@cnt @options\endcsname{#1}%
 \expandafter\def\csname nowfnt@article@\the\c@nowfnt@article@cnt @folder\endcsname{#2}%
 \expandafter\def\csname nowfnt@article@\the\c@nowfnt@article@cnt @file\endcsname{#2}%
 \externaldocument[article-\csname nowfnt@article@\the\c@nowfnt@article@cnt @file\endcsname-]%????
                  {\csname nowfnt@article@\the\c@nowfnt@article@cnt @folder\endcsname/\csname nowfnt@article@\the\c@nowfnt@article@cnt @file\endcsname}%
 }


\newcommand\compilearticles{%
 \nowfnt@write@luacompile
 \nowfnt@run@luacompile
 }

\newwrite\nowfnt@write
\newcommand\nowfnt@write@luacompile{%
 \immediate\openout\nowfnt@write=compile-wrapper-chapters.lua
 \immediate\write\nowfnt@write{kpse.set_program_name("luatex")}
 \immediate\write\nowfnt@write{dofile(kpse.find_file("now-compile.lua"))}
 \c@nowfnt@tempcnt=0\relax
 \whiledo{\c@nowfnt@tempcnt<\c@nowfnt@article@cnt}
  {\stepcounter{nowfnt@tempcnt}%
   \ifnum\c@nowfnt@tempcnt=1
    \immediate\write\nowfnt@write{now_compile_article("\csname nowfnt@article@\the\c@nowfnt@tempcnt @folder\endcsname","\csname nowfnt@article@\the\c@nowfnt@tempcnt @file\endcsname-nowchapter")}%
    \immediate\write\nowfnt@write{now_compile_article("\csname nowfnt@article@\the\c@nowfnt@tempcnt @folder\endcsname","\csname nowfnt@article@\the\c@nowfnt@tempcnt @file\endcsname-nowechapter")}%
   \else
    \immediate\write\nowfnt@write{%
        now_compile_article(%
         \string"\csname nowfnt@article@\the\c@nowfnt@tempcnt @folder\endcsname\string",%
         \string"\csname nowfnt@article@\the\c@nowfnt@tempcnt @file\endcsname-nowchapter\string",%
         \string"\csname nowfnt@article@\the\numexpr\c@nowfnt@tempcnt-1 @folder\endcsname\string",%
         \string"\csname nowfnt@article@\the\numexpr\c@nowfnt@tempcnt-1 @file\endcsname-nowchapter\string")}%
    \immediate\write\nowfnt@write{%
        now_compile_article(%
         \string"\csname nowfnt@article@\the\c@nowfnt@tempcnt @folder\endcsname\string",%
         \string"\csname nowfnt@article@\the\c@nowfnt@tempcnt @file\endcsname-nowechapter\string",%
         \string"\csname nowfnt@article@\the\numexpr\c@nowfnt@tempcnt-1 @folder\endcsname\string",%
         \string"\csname nowfnt@article@\the\numexpr\c@nowfnt@tempcnt-1 @file\endcsname-nowechapter\string")}%
   \fi}
   \immediate\closeout\nowfnt@write
   }

\newcommand\nowfnt@run@luacompile{}

\if@nowfntwrapper
\renewcommand\nowfnt@run@luacompile{%
  \ifnum\pdfshellescape=1
  % Yes, enabled
   \immediate\write18{texlua compile-wrapper-chapters.lua}
  \else
  % No, disabled
  \ClassWarning{nowfnt}{shell-escape is not enabled. I can't compile the articles}
  \fi}
\fi

\newcommand\nowfnt@includearticles{%
 \c@nowfnt@tempcnt=0\relax
 \whiledo{\c@nowfnt@tempcnt<\c@nowfnt@article@cnt}
  {%
   \stepcounter{nowfnt@tempcnt}%
   \expandafter\let\expandafter\nowfnt@tmpoptions\csname nowfnt@article@\the\c@nowfnt@tempcnt @options\endcsname
   %\show\nowfnt@tmpoptions
   \expandafter\nowinclude\expandafter[\nowfnt@tmpoptions]
    {%
     \csname nowfnt@article@\the\c@nowfnt@tempcnt @folder\endcsname/%
     \csname nowfnt@article@\the\c@nowfnt@tempcnt @file\endcsname
     \if@nowfntwrapperbook
      -nowchapter%
     \else
      -nowechapter%
     \fi
    }%
  }}

\let\createjournal\nowfnt@includearticles

\ifcsname tikzexternalrealjob\endcsname \else
\ifbool{@nowfntplain}
{\AtBeginDocument{\maketitle}}
{
 \AtBeginDocument{\if@nowfntchapter\else\frontmatter \maketitle \clearpage\tableofcontents\fi\mainmatter}
 \AtBeginDocument{\addtocontents{toc}{\protect\thispagestyle{empty}}}
 }
\fi
\endinput
