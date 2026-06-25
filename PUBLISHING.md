# Publishing & rollout

Concrete steps for getting this package onto PyPI, reserving the npm name for later,
deciding where the SDK/docs live, and the specific distribution playbook for getting
the first wave of developer adoption.

## 1. PyPI (the actual package)

The bare name `n00dles` was unclaimed but PyPI's typosquat-prevention check rejected
it on upload as "too similar to an existing project" — there's already a `noodles`
package (a workflow engine), and PyPI treats `0`/`o` as confusable characters. The
distribution name is **`get-n00dles`** instead (checked unclaimed via
`pypi.org/pypi/get-n00dles/json` → 404). The Python import is unaffected either way —
it's still `import n00dles`; only `pip install get-n00dles` differs from what you'd
guess.

**v0.1.0 was published manually** using a PyPI API token (`__token__` / token from
`.env`, never committed — see `.gitignore`) via `twine upload`. That's fine for a
one-off, but every future release re-doing that manual token dance doesn't scale and
keeps a long-lived token sitting in a local `.env`. Switch to CI-driven publishing
via a **pending trusted publisher** (no API tokens at all, ever — PyPI lets you
register a trusted publisher for a project that already exists, same flow):

1. Log into [pypi.org](https://pypi.org) with whichever account currently owns the
   `get-n00dles` project (the one whose token was used for the manual v0.1.0 upload).
2. On the project page → **Publishing** → "Add a new publisher":
   - Owner: `n00dlehouse`
   - Repository: `n00dles-py`
   - Workflow filename: `publish.yml`
   - Environment name: `pypi`
3. `.github/workflows/publish.yml` (in this repo) runs on tag push (`v*`), builds with
   `python -m build`, and uploads via
   [`pypa/gh-action-pypi-publish`](https://github.com/pypa/gh-action-pypi-publish) —
   no secret token in GitHub at all once the trusted publisher is registered; trust is
   established via OIDC between GitHub Actions and PyPI directly.
4. To ship a release from then on: bump `version` in `pyproject.toml`, commit, then
   `git tag v0.1.1 && git push origin v0.1.1`. The workflow builds and publishes
   automatically — no token, no manual `twine upload`.
5. Once the trusted publisher is registered and confirmed working, delete the API
   token from PyPI account settings (**pypi.org/manage/account/token/**) and from the
   local `.env` — it no longer needs to exist.

**Before the first real publish:**

- Decide on the PyPI account/org that owns the project (personal account vs. a
  dedicated `n00dlehouse` PyPI org — PyPI orgs are free and avoid a single person's
  account being a bus-factor-of-one for the project's primary distribution channel).
- Double check `pyproject.toml`'s `Repository`/`Issues`/`Documentation` URLs are live
  before publishing — broken links on a fresh PyPI page are a bad first impression.

## 2. npm — reserve the name, don't publish a package

Per the agreed scope, there's no JS/TS implementation yet. But `n00dles` is also free
on npm (`registry.npmjs.org/n00dles` → 404) and `@n00dles/core` is free too. Two
options, in order of recommendation:

- **Do nothing for now.** Names aren't guaranteed to stay free, but squatting an empty
  package mostly just adds noise to npm search results pointing at nothing — low value.
- **If you want insurance against someone else grabbing it**, publish a single empty
  placeholder (`npm init`, version `0.0.0`, a README that says "this is reserved for
  the future n00dles JS SDK — see github.com/n00dlehouse/n00dles for the Python
  package available today") under the same GitHub account. Costs nothing, prevents
  squatting, doesn't promise functionality that doesn't exist.

## 3. Where things live

- **Code + issues + PRs:** `github.com/n00dlehouse/n00dles` — the source of truth.
  Don't split development across multiple repos this early.
- **Docs:** the architecture doc specifies `docs.n00dles.com` as a dedicated docs
  site. For v0.1, that's premature — the README in this repo plus the marketing
  site's `/docs` pages (already built, already live) cover the same ground without
  standing up and maintaining a third surface. Revisit a dedicated docs site (Mintlify,
  Docusaurus, or a Next.js app like the marketing site already is) once the API
  surface is stable enough that docs aren't changing weekly.
- **Package page:** PyPI's auto-generated page (rendered from this README) is enough
  at this stage — it already links to the GitHub repo and the marketing site.

## 4. Getting the first wave of developer adoption

The single most important early milestone, per the architecture doc, is one real
developer running one real pipeline in production and telling someone about it.
Marketing spend doesn't substitute for that — sequencing matters:

**Before posting anywhere:**
- Tag a real `v0.1.0` release with release notes (GitHub Releases, not just a git tag).
- Make sure `pip install get-n00dles` → quickstart → first pipeline genuinely takes
  under 5 minutes, copy-paste, no surprises. This is the thing being advertised — it
  has to actually be true.
- Seed the repo with 5-10 GitHub issues labeled `good first issue` so the first wave
  of interested contributors has something concrete to do.

**Distribution, roughly in order:**
1. **Show HN** (`news.ycombinator.com`) with a title that states the concrete claim,
   not a slogan — e.g. "n00dles – multi-agent orchestration in ~10 lines, wraps
   litellm" outperforms vague taglines. Post Tuesday–Thursday, US morning hours.
2. **r/LocalLLaMA and r/MachineLearning** — technical audiences that actually try
   things, not just upvote. Lead with the code sample, not the pitch.
3. **Twitter/X** — a short thread showing the LangChain-vs-n00dles line-count
   comparison (already the exact narrative on the marketing site's homepage) is a
   format that travels well. Tag people who've publicly complained about LangChain
   boilerplate; that complaint is your entire audience.
4. **A comparison blog post** — "Migrating from LangChain to n00dles" or "Why we
   wrapped litellm instead of writing 5 provider SDKs" both work as long-tail SEO
   that keeps paying off for months after the initial launch spike fades. Cross-post
   to dev.to and Hashnode in addition to the marketing site's `/blog`.
5. **Awesome-list PRs** — `awesome-llm`, `awesome-llmops`, and similar curated lists
   are low-effort, durable backlinks that developers actually browse when evaluating
   options.
6. **Discord** — stand up a server (or a channel in an existing AI-builders Discord)
   *before* the Show HN post, not after, so day-one visitors have somewhere to ask
   questions in real time instead of going quiet.

**What not to do:** don't post to more than 2-3 channels on the same day — a flat
launch with no follow-up content is worse than a staggered one, since the algorithm-
driven channels (HN, Reddit) reward sustained engagement more than initial volume.
Hold the comparison blog post and the Twitter thread for a few days after Show HN so
there's a second wave instead of one spike.
