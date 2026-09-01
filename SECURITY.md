# Security Policy

## Supported Versions

This project does not yet maintain multiple supported release branches.
Security fixes are made against the latest code on `main`; please make sure
you're using the most recent release before reporting an issue.

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, report it privately by emailing **github@myoddweb.com**, or by
using GitHub's [private vulnerability reporting](https://github.com/FFMG/neural-network/security/advisories/new)
for this repository.

Please include:

* A description of the issue and its potential impact.
* Steps to reproduce (a minimal repro is ideal — e.g. a malformed model
  file or a specific `NeuralNetworkOptions` configuration).
* The affected version/commit.

You should expect an initial response within a few days. If the issue is
confirmed, a fix will be prepared and a GitHub Security Advisory published
once a patch is available; you'll be credited unless you'd prefer otherwise.

## Scope

This is a local C++/Python library with no network-facing components and
zero external runtime dependencies for the core library. The main area of
security relevance is **deserialization of untrusted model files** via
`NeuralNetworkSerializer::load()`: as with any binary deserializer, loading
a malformed or maliciously crafted `.nn` file is not guaranteed to be safe.
Do not load model files from sources you don't trust.
