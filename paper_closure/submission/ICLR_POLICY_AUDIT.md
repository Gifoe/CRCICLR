# ICLR 2027 Policy Audit

Audit date: 2026-08-29 (Asia/Shanghai)

Audit terminal: POLICY_BASIS_CONFIRMED_ICLR_2027; SUBMISSION_PACKAGE_NOT_YET_VERIFIED

## Scope and authority

This is a policy and template-provenance audit for an initial ICLR 2027
submission. It does not certify the current manuscript or supplementary package
as submission-ready. No manuscript, experiment, OpenReview record, or submission
state was modified.

The current official year is available. The applicable page is
AuthorGuidelines (plural), not the older AuthorGuide path. The latter returned
HTTP 404 on the audit date; this path change must not be misread as absence of
2027 rules.

Final decisions below use only official ICLR pages, the official ICLR template
archive, and the official OpenReview venue/form API. The installed local ICLR
venue note is excluded as a policy source because it is internally
contradictory: it labels ICLR 2026 double-blind in one section and later claims
that submissions are not anonymous. It is not reliable enough for any anonymity
decision.

## Submission decision table

| Item | Authoritative ICLR 2027 rule | Required closure action | Status |
| --- | --- | --- | --- |
| Initial main text | At most 9 pages | Build and count the review PDF at 9 main-text pages or fewer | Confirmed rule; PDF not yet checked |
| Discussion/rebuttal and camera-ready main text | Limit increases to 10 pages | Do not use the extra page in the initial submission | Confirmed rule |
| References | Unlimited; excluded from the page limit | Put the bibliography after all counted main-text content | Confirmed rule |
| Appendix | Unlimited pages after the bibliography; reviewers need not read it | Keep nonessential evidence in the appendix and do not rely on it for a core claim | Confirmed rule |
| Supplementary text | A single paper-plus-supplement PDF is encouraged; supplementary text follows the references and is clearly marked, for example as an appendix | Prefer one review PDF unless a separate supplement is operationally necessary | Confirmed rule |
| Separate appendix/supplement | Appendix may instead be a separate supplementary file; deadline is the full-paper deadline | If separate, package it under the OpenReview constraints below | Confirmed rule |
| Review anonymity | ICLR 2027 is double-blind; identity in either main text or supplementary material causes desk rejection | Produce an anonymous PDF and an anonymous supplementary package | Hard blocker until checked |
| Author block | The official style hides the author block while \iclrfinalcopy remains commented and prints “Anonymous authors / Paper under double-blind review” | Keep \iclrfinalcopy commented for submission; use anonymous placeholders in the review-build source as a conservative leakage control | Confirmed rule; source not yet checked |
| Current template | Authors must use the current ICLR 2027 LaTeX style; modifying style/formatting may cause rejection | Integrate the exact official 2027 files recorded below without edits | Confirmed rule; not yet integrated |
| AI-use disclosure | Required both in the paper and in the OpenReview form; the paper section is excluded from the page limit | Add a truthful, author-verified AI use statement before submission | Hard blocker |
| Ethics statement | Recommended when the work raises ethics issues, including human-subject research; excluded from page limit and at most one page | Verify dataset consent/ethics facts and add a sourced statement if applicable | Unresolved manuscript item |
| Reproducibility statement | Strongly recommended; paragraph-long, before references, excluded from page limit | Add a statement pointing to reproducibility material rather than duplicating it | Unresolved manuscript item |
| Acknowledgements | Excluded from page count, but submission anonymity still governs | Omit or anonymize identity-revealing acknowledgements/funding in the review version; restore only when permitted | Hard anonymity control |
| Main PDF upload | OpenReview form accepts PDF, maximum 50 MB | Check final file type and size | Not yet checked |
| Supplement upload | One self-contained ZIP or PDF, maximum 100 MB; visible to reviewers and the public throughout and after review; all material must be anonymized | Run archive-content and metadata anonymity checks before upload | Hard blocker until checked |

## Official sources and retrieval record

Retrieval was performed from the project server in read-only mode. The local
calendar date was 2026-08-29 (Asia/Shanghai); the exact captures below were made
on 2026-08-28 UTC. SHA256 values for HTML and JSON are point-in-time response
hashes, not checksums published by ICLR.

| Official source | Retrieval (UTC) | HTTP | Point-in-time SHA256 | Decision use |
| --- | --- | ---: | --- | --- |
| https://iclr.cc/Conferences/2027/AuthorGuidelines | 2026-08-28 18:17:18 | 200 | 161192145e212c882f262d75092a9768e33682e2b5d2aea7d30e13dc6869baf6 | Primary submission, anonymity, page-limit, appendix, supplementary, and timeline rules |
| https://iclr.cc/Conferences/2027/CallForPapers | 2026-08-28 18:17:19 | 200 | cce4d100ad67214eb32cff00f7eade0525c8ffff67d923e2cbae6726774e87f0 | Confirms the 2027 call, double-blind review, and the AuthorGuidelines link |
| https://iclr.cc/Conferences/2027/AIPolicyForAuthors | 2026-08-28 18:17:21 | 200 | bdfa4118c924c541338f2657bd47ae9602e4399a7a4699fbddbd9861a4d51708 | Mandatory AI disclosure and author responsibility |
| https://api2.openreview.net/groups?id=ICLR.cc/2027/Conference | 2026-08-28 18:17:22 | 200 | 7ae32e80015a877ee507ef2d43dca23820cc1f28ad40e35f548df8cc04911074 | Confirms the official ICLR 2027 venue and submission endpoint |
| https://api2.openreview.net/invitations?id=ICLR.cc/2027/Conference/-/Submission | 2026-08-28 18:17:22 | 200 | 4592de0b68b68425c5e1c9c3f15a4d6f139c659bf7efae9d358dd83ac07b3299 | Current upload fields, size limits, supplementary visibility/anonymity, author-profile and AI-form requirements |
| https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip | 2026-08-28 18:12:51 | 200 | 0d940dfa9398ae99a18f24a85a8a683f367204b6af6d17d2899e60a67102529e | Official LaTeX template distribution |

The official OpenReview human-facing venue is:
https://openreview.net/group?id=ICLR.cc/2027/Conference .

## Page limit and document order

The operative initial-submission rule is 9 pages of main text or fewer. The
official guide explicitly says that the limit increases to 10 pages only during
discussion/rebuttal and for camera-ready. Papers whose main text exceeds the
applicable limit are desk-rejected.

The safe review-version order is:

1. title, anonymous author block, abstract, and counted main text;
2. required AI use statement;
3. optional ethics statement and recommended reproducibility statement;
4. references;
5. appendix/supplementary text.

The guide excludes references from the limit and permits unlimited bibliography
pages. It permits unlimited appendix pages after the bibliography, while warning
that reviewers are not required to read them. Therefore, a central claim,
definition, primary result, or fact necessary to understand validity cannot be
delegated only to the appendix.

The official template states that the AI use statement is required, excluded
from the page limit, and should not exceed one page. The optional ethics
statement is also excluded and should not exceed one page. The reproducibility
statement is recommended, paragraph-long, and excluded. The FAQ says
acknowledgements are excluded as well, but this exemption does not override the
double-blind rule.

Two FAQ lines use the awkward phrase “identical with the submission version
(10 pages)” for rebuttal and camera-ready. This does not change the initial
limit: the primary formatting paragraph unambiguously specifies 9 pages at
initial submission and an increase to 10 pages for those later versions. The
conservative and internally consistent rule is therefore 9 initial / 10 later.

## Double-blind and author-block controls

ICLR 2027 states that all submitted papers must be anonymous and that revealing
author identity in the main text or supplementary material results in desk
rejection. The OpenReview form still requires the real authors and their
profiles. Those form fields are submission metadata; they do not authorize
author names, affiliations, email addresses, acknowledgements, or identifying
project links in the review PDF or supplementary package.

The official template implements review mode as follows:

- \iclrfinalcopy is commented for submission and is uncommented only for the
  camera-ready version.
- In review mode, the style renders “Anonymous authors” and “Paper under
  double-blind review” instead of the contents of the LaTeX author block.
- The sample template warns that authors must not appear in the submitted
  version and that non-anonymous submissions are rejected without review.

For this closure package, the lower-risk implementation is to place anonymous
placeholders in the review-build author source as well as keeping
\iclrfinalcopy commented. Although the style hides an ordinary author macro,
real names in a source tree can leak through supplementary archives, build
artifacts, comments, or an accidental final-mode build.

Related arXiv papers do not by themselves break anonymity. If cited, they must be
cited in the third person. An almost identical arXiv version is allowed under
the stated policy so long as the submission does not explicitly point to it in
a way that deanonymizes the authors.

The final anonymity check must cover more than visible prose:

- PDF author/title metadata and embedded attachments;
- acknowledgements, funding, ethics/IRB wording, author contributions, and
  self-identifying first-person references;
- filenames, file paths, figure metadata, comments, and tracked-change residue;
- code headers, package metadata, licenses, environment files, experiment
  dashboards, repository remotes, Git history, commit authors, and account IDs;
- repository, data, demo, and video URLs, including redirects and landing pages.

These checks are operational controls derived from the official no-identity
rule. They are not claims that ICLR enumerates every listed metadata field.

## Supplementary material, code, repositories, and links

The author guide encourages one combined paper-plus-supplementary-text PDF.
Supplementary text follows the references and is clearly marked as an appendix.
The FAQ also permits a separate appendix in the supplementary upload. The
supplementary deadline is the same as the full-paper deadline.

Code may be uploaded as supplementary material and is encouraged for
replicability, but reviewers are encouraged rather than required to review any
supplement. The official OpenReview form is stricter about packaging:

- all supplementary material must be self-contained in one file;
- accepted formats are ZIP or PDF;
- maximum size is 100 MB;
- the material is visible to reviewers and the public throughout and after the
  review period;
- every item must be anonymized.

The official FAQ lists three code-sharing routes:

1. anonymize the code, ZIP it, and upload it as supplementary material;
2. use an anonymous repository and put its link in the paper;
3. after discussion opens, send reviewers and area chairs a comment linking to
   an anonymous repository, with the comment restricted to those readers.

The first two routes make the code public with the paper and reviews/comments.
The third can keep it visible only to the paper's reviewers and area chairs.
Demonstration links, including video links, must be completely anonymous. Their
hosting site must not track visitors in a way that could reveal reviewer
identity; the guide says such a link puts the submission at risk of rejection.

## Official template provenance

Official archive:
https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip

- Retrieval: 2026-08-28T18:12:51.4251123Z
- Content-Type: application/zip
- Last-Modified: Tue, 28 Jul 2026 18:53:19 GMT
- ETag: "801ff08900708c20-99b4-657b05a2ca5c0"
- Size: 39,348 bytes
- SHA256: 0d940dfa9398ae99a18f24a85a8a683f367204b6af6d17d2899e60a67102529e

| Archive member | Bytes | SHA256 |
| --- | ---: | --- |
| iclr2027/fancyhdr.sty | 20,521 | b56ec4434b9f4607529a4b23dc68ad8d4b94f1f631c8cddaf7da78140d53a5ea |
| iclr2027/iclr2027_conference.bib | 629 | cdd86e7d4c31854dcf2145871657c944588a6d44c3b72e160ff4baa8df1a52fb |
| iclr2027/iclr2027_conference.bst | 26,973 | 2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5 |
| iclr2027/iclr2027_conference.sty | 9,025 | 797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6 |
| iclr2027/iclr2027_conference.tex | 19,740 | 03e556d6e5593e498fd39f262ec2d184ebfe8693cbf47fb7cea3402d8d5166ac |
| iclr2027/math_commands.tex | 12,284 | 90473c4d0542070db244cea73ef962d6cddc5b2a746757e6a40ddf5fdfb90ba9 |
| iclr2027/natbib.sty | 45,154 | 88bc70c0e48461934cab5b2accef06b74a8b3ac45ad03ccd3f2a6b7e0d6d530d |

The archive was downloaded only to temporary storage for this audit and was not
added to the repository. The retrieval hash above is an audit-generated
point-in-time hash; the official page does not publish a checksum or detached
signature. Re-download from the same official URL and compare before the final
submission build.

The official sample says submissions must use iclr2027_conference.sty and
iclr2027_conference.bst, that iclr2027_conference.tex may be used as the writing
shell, and that authors must use the current files rather than an earlier
version. It warns that tweaking the style files may be grounds for rejection and
specifically prohibits changes to the text rectangle and font sizes. It also
requires US Letter paper size and numbered pages.

The sample text mentions an iclr2027_conference.pdf instruction file, but that
PDF is not present in the downloaded official ZIP. This packaging inconsistency
does not prevent use of the distributed TEX/STY/BST files, but no nonexistent
PDF should be invented or cited as an audited artifact.

## AI-use disclosure

ICLR 2027 requires authors to state how generative AI/LLM tools were used both
in the paper and in the OpenReview submission form. The manuscript section does
not count toward the page limit. The OpenReview form warns that failure to
include the mandatory paper section can lead to desk rejection.

The official AI policy requires disclosure for uses including research
methodology or experiment feedback, hypothesis refinement, theoretical claims
or proofs, method implementation, data cleaning/reformatting, translation,
qualitative analysis, and result interpretation. It recommends disclosure for,
among other uses, code editing, paper drafting/editing, literature search and
summarization, reference formatting, figures, brainstorming, title/keyword
suggestions, and paper structuring.

The exact statement for this project is unresolved. It must be written and
approved by the authors from the actual project-use ledger; this audit must not
guess which research, code, analysis, retrieval, and writing tasks were
AI-assisted. The statement must also record that authors reviewed the assisted
work and remain responsible for the final content. Material falsehood,
plagiarism, or misrepresentation produced by an AI tool remains the authors'
Code of Ethics responsibility and may lead to desk rejection.

## 2027 timeline and author-administration risks

- Abstract deadline: 2026-09-18, 11:59 PM Anywhere on Earth.
- Full paper and supplementary deadline: 2026-09-25, 11:59 PM Anywhere on
  Earth.
- No new authors may be added after the abstract deadline. Author order may be
  changed through the full-paper deadline.
- All authors need current OpenReview profiles.
- No author may appear on more than 20 submissions.
- Reciprocal-reviewing requirements apply. At least one author must be
  registered for the required reviewing role unless the program chairs grant
  the applicable exemption; authors on three or more submissions have an
  additional six-review obligation under the published rule.

The final author list, author order, profiles, submission quotas, and
reciprocal-reviewer eligibility were not examined in this policy-only audit.
They are administrative desk-rejection risks and require a separate author
check before the abstract deadline.

## Unresolved items and submission blockers

1. No final review PDF has been compiled and counted against the 9-page initial
   main-text limit.
2. The closure mirror did not contain main.tex or the ICLR 2027 template files
   at the time of this audit. The eventual integration must be hash-checked
   against the official package above.
3. No rendered-format audit has yet verified US Letter size, page numbering,
   unchanged margins/fonts, references, appendix order, or absence of style
   modifications.
4. No PDF or supplementary anonymity scan has been run. Identity leakage in
   either is a desk-rejection condition.
5. The exact, truthful AI use statement and matching OpenReview-form responses
   have not been drafted or author-approved.
6. The human-EEG ethics statement, consent/IRB provenance, data-use terms, and
   any anonymity-safe wording have not been checked against the original
   dataset sources.
7. The reproducibility statement and anonymous artifact route have not been
   selected. Reviewers are not required to inspect appendices or code, so core
   support cannot depend on them.
8. The final author/profile/quota/reciprocal-reviewing audit remains open.
9. Camera-ready upload instructions are scheduled for mid-February 2027 and are
   not yet available; current review-stage rules must not be extrapolated into
   missing camera-ready administrative details.
10. The official template archive has no publisher-posted checksum. Its
    point-in-time hash must be revalidated if ICLR changes the asset.

## Readiness verdict

The governing 2027 policy is available and sufficiently clear to build the
initial manuscript. The package is not submission-ready merely because the
policy has been identified. Submission readiness requires the official template
to be integrated unchanged, a rendered 9-page review PDF, an author-approved AI
statement, and successful anonymity and supplementary-package audits.
