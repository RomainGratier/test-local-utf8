# Validation Tools

Auto-synced from AI worker. Used by AI agent to validate .zoey file updates.

## Files

- `codefolio-validator.ts`: Validates codefolio.json structure
- `knowledge-validation.ts`: Validates knowledge.yaml structure
- `validate.js`: Ready-to-run validation script
- `node_modules/js-yaml`: Dependency (copied from worker)

## Usage

Quick validation (recommended):
```bash
cd .zoey/validators && npm run validate
```

Manual validation from TypeScript:
```typescript
import { validateCodefolioStructure } from './.zoey/validators/codefolio-validator.ts';
import { validateKnowledgeYaml } from './.zoey/validators/knowledge-validation.ts';
// ... use validators
```

These files are overwritten on each AI Updater run to ensure version sync.
