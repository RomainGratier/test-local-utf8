/**
 * Shared Codefolio JSON validation utilities
 * This file is designed to work in Node.js, Deno, and Browser environments
 *
 * IMPORTANT: This is the canonical validation - keep it in sync across:
 * - supabase/functions/_shared/codefolio-validator.ts (backend - Deno)
 * - workers/ai-agent-worker/src/codefolio-validator.ts (worker - Node)
 * - src/lib/codefolio-validator.ts (frontend - Browser)
 */
export const VALID_SECTION_TYPES = [
    "text",
    "code",
    "git_diff",
    "video",
    "image",
    "link",
    "iframe",
    "labels",
    "project",
    "experience",
    "project_showcase",
    "github_issue",
    "github_pr",
    "github_commit_story",
    "skills_showcase",
    "achievement",
    "markdown",
    "header",
    "paragraph",
];
/**
 * Validates the structure of a codefolio.json file
 * This is the canonical validation function used everywhere
 */
export function validateCodefolioStructure(codefolio) {
    const errors = [];
    if (typeof codefolio !== "object" || codefolio === null) {
        errors.push("Codefolio must be an object");
        return { valid: false, errors };
    }
    const cf = codefolio;
    // Required fields check
    const required = ["version", "title", "description", "sections"];
    for (const field of required) {
        if (!(field in cf)) {
            errors.push(`Missing required field: ${field}`);
        }
    }
    // Type validation
    if (cf.version && typeof cf.version !== "string") {
        errors.push("version must be a string");
    }
    if (cf.title && typeof cf.title !== "string") {
        errors.push("title must be a string");
    }
    if (cf.description && typeof cf.description !== "string") {
        errors.push("description must be a string");
    }
    // Sections must be an array
    if (!Array.isArray(cf.sections)) {
        errors.push("sections must be an array");
        return { valid: errors.length === 0, errors };
    }
    // Validate each section
    for (let i = 0; i < cf.sections.length; i++) {
        const section = cf.sections[i];
        if (typeof section !== "object" || section === null) {
            errors.push(`Section at index ${i} must be an object`);
            continue;
        }
        const s = section;
        // Required section fields
        if (!s.id) {
            errors.push(`Section at index ${i} missing required field: id`);
        }
        else if (typeof s.id !== "string") {
            errors.push(`Section at index ${i}: id must be a string`);
        }
        if (!s.type) {
            errors.push(`Section at index ${i} missing required field: type`);
        }
        else if (typeof s.type !== "string") {
            errors.push(`Section at index ${i}: type must be a string`);
        }
        else if (!VALID_SECTION_TYPES.includes(s.type)) {
            errors.push(`Section at index ${i}: invalid section type "${s.type}". Must be one of: ${VALID_SECTION_TYPES.join(", ")}`);
        }
        if (!s.title) {
            errors.push(`Section at index ${i} missing required field: title`);
        }
        else if (typeof s.title !== "string") {
            errors.push(`Section at index ${i}: title must be a string`);
        }
        if (s.content === null || s.content === undefined) {
            errors.push(`Section at index ${i} missing required field: content`);
        }
        else if (typeof s.content !== "object" || Array.isArray(s.content)) {
            errors.push(`Section at index ${i}: content must be an object (not array or null)`);
        }
    }
    return { valid: errors.length === 0, errors };
}
/**
 * Validates and returns typed codefolio or throws with detailed errors
 */
export function validateAndParseCodefolio(input) {
    const result = validateCodefolioStructure(input);
    if (!result.valid) {
        throw new Error(`Invalid codefolio structure:\n${result.errors.join("\n")}`);
    }
    return input;
}
/**
 * Quick check for testing - returns boolean
 */
export function isValidCodefolio(input) {
    return validateCodefolioStructure(input).valid;
}
