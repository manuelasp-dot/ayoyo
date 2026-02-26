# CLAUDE.md — AI Assistant Guide for Ayoyo

## Project Overview

Ayoyo is a newly initialized repository. This file serves as the canonical reference for AI assistants (e.g., Claude Code) working in this codebase.

## Repository Structure

```
ayoyo/
├── CLAUDE.md          # This file — AI assistant guide and project conventions
└── .git/              # Git repository metadata
```

> **Note:** This project is in its initial state. Update this section as the directory structure evolves.

## Getting Started

### Prerequisites

- Git

### Setup

```bash
git clone <remote-url>
cd ayoyo
```

## Development Workflow

### Branch Naming

- Feature branches: `feature/<description>`
- Bug fixes: `fix/<description>`
- Claude Code branches: `claude/<description>-<session-id>`

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in imperative mood (e.g., "Add", "Fix", "Update", "Remove")
- Keep the subject line under 72 characters
- Add a body for non-trivial changes explaining *why*, not just *what*

### Code Style

- Follow the conventions of whichever language/framework is adopted
- Prefer readability over cleverness
- Keep functions small and focused
- Name variables and functions descriptively

## Testing

> **TODO:** Add testing framework and conventions once the project stack is chosen.

## CI/CD

> **TODO:** Add CI/CD pipeline details once configured.

## Key Conventions for AI Assistants

1. **Read before writing** — Always read existing files before modifying them.
2. **Minimal changes** — Only change what is necessary to accomplish the task. Avoid unnecessary refactors or additions.
3. **No over-engineering** — Keep solutions simple. Don't add abstractions, error handling, or features beyond what's requested.
4. **Preserve existing patterns** — Match the style, structure, and conventions already present in the codebase.
5. **Test your changes** — Run existing tests after making changes. Don't break what already works.
6. **Don't commit secrets** — Never commit `.env` files, API keys, credentials, or other sensitive data.
7. **Update this file** — When adding significant new tooling, conventions, or architectural decisions, update this CLAUDE.md to reflect them.
