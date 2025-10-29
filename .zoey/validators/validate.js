#!/usr/bin/env node
/**
 * Validation script for .zoey files
 * Usage: npm run validate (from .zoey/validators directory)
 */

import { readFileSync } from 'fs';
import { validateCodefolioStructure } from './codefolio-validator.js';
import { validateKnowledgeYaml } from './knowledge-validation.js';

console.log('🔍 Validating .zoey files...\n');

let hasErrors = false;

// Validate codefolio.json
try {
    const codefolio = JSON.parse(readFileSync('../codefolio.json', 'utf-8'));
    const result = validateCodefolioStructure(codefolio);
    if (!result.valid) {
        console.error('❌ Codefolio validation failed:');
        result.errors.forEach(err => console.error('  -', err));
        hasErrors = true;
    } else {
        console.log('✅ Codefolio validation passed');
    }
} catch (error) {
    console.error('❌ Codefolio error:', error.message);
    hasErrors = true;
}

// Validate knowledge.yaml
try {
    const knowledge = readFileSync('../knowledge.yaml', 'utf-8');
    const result = validateKnowledgeYaml(knowledge);
    if (!result.valid) {
        console.error('\n❌ Knowledge validation failed:');
        result.errors.forEach(err => console.error('  -', err));
        hasErrors = true;
    } else {
        console.log('✅ Knowledge validation passed');
    }
} catch (error) {
    console.error('\n❌ Knowledge error:', error.message);
    hasErrors = true;
}

if (hasErrors) {
    console.error('\n❌ Validation failed - please fix errors above\n');
    process.exit(1);
} else {
    console.log('\n✅ All validations passed!\n');
}
