import { load as yamlLoad } from "js-yaml";
/**
 * Validates the structure and content of a knowledge.yaml file
 */
export function validateKnowledgeYaml(content) {
    const errors = [];
    const warnings = [];
    const isRecord = (val) => typeof val === "object" && val !== null && !Array.isArray(val);
    const isArrayOfStrings = (val) => Array.isArray(val) && val.every((v) => typeof v === "string");
    try {
        // Parse YAML
        const parsedUnknown = yamlLoad(content);
        if (!isRecord(parsedUnknown)) {
            return {
                valid: false,
                errors: ["Invalid YAML: content is not an object"],
                warnings,
            };
        }
        const parsed = parsedUnknown;
        // Required top-level fields
        if (!("overview" in parsed) || typeof parsed.overview !== "string") {
            errors.push("Missing or invalid 'overview' field - must describe the challenge purpose");
        }
        if (!("requirements" in parsed) || !isRecord(parsed.requirements)) {
            errors.push("Missing 'requirements' section - must define functional requirements");
        }
        else {
            const req = parsed.requirements;
            const core = req.core_features;
            if (!isArrayOfStrings(core) || core.length === 0) {
                errors.push("Missing 'requirements.core_features' - must list required features");
            }
        }
        if (!("constraints" in parsed) || !isRecord(parsed.constraints)) {
            errors.push("Missing 'constraints' section - must define technical constraints");
        }
        const ac = parsed.acceptance_criteria;
        if (!Array.isArray(ac) || ac.length === 0) {
            errors.push("Missing 'acceptance_criteria' - must define how success is validated");
        }
        else {
            // Validate each acceptance criterion has required fields
            ac.forEach((criterionUnknown, index) => {
                if (!isRecord(criterionUnknown)) {
                    errors.push(`Acceptance criterion ${index + 1}: invalid item`);
                    return;
                }
                const criterion = criterionUnknown;
                if (typeof criterion.criteria !== "string") {
                    errors.push(`Acceptance criterion ${index + 1}: missing 'criteria' field`);
                }
                if (!isArrayOfStrings(criterion.tests) ||
                    criterion.tests.length === 0) {
                    errors.push(`Acceptance criterion ${index + 1}: missing or empty 'tests' array`);
                }
            });
        }
        // Warnings for optional but recommended fields
        if (!("evaluation_focus" in parsed)) {
            warnings.push("Consider adding 'evaluation_focus' to clarify what you're assessing");
        }
        if (!("ai_agent_guidance" in parsed)) {
            warnings.push("Consider adding 'ai_agent_guidance' to help the AI understand coding expectations");
        }
        const reqObj = isRecord(parsed.requirements)
            ? parsed.requirements
            : undefined;
        if (reqObj && isArrayOfStrings(reqObj.core_features)) {
            const hasPlaceholder = reqObj.core_features.some((f) => f.toLowerCase().includes("define"));
            if (hasPlaceholder) {
                warnings.push("Some requirements contain placeholder text (e.g., 'Define core feature') - update with actual requirements");
            }
        }
        if (errors.length > 0) {
            return { valid: false, errors, warnings };
        }
        return {
            valid: true,
            errors: [],
            warnings,
            parsed: parsed,
        };
    }
    catch (error) {
        return {
            valid: false,
            errors: [
                `Failed to parse YAML: ${error instanceof Error ? error.message : String(error)}`,
            ],
            warnings,
        };
    }
}
/**
 * Checks if a user request is within the scope of the knowledge base requirements
 */
export function validateRequestScope(userRequest, knowledge) {
    const requestLower = userRequest.toLowerCase();
    // Extract all defined features/requirements
    const allFeatures = [
        ...(knowledge.requirements.core_features || []),
        ...(knowledge.requirements.nice_to_have || []),
    ].map((f) => f.toLowerCase());
    // Check if request mentions features not in requirements
    const potentialOutOfScopeRegex = [
        /\badd\s+(a\s+)?new\b/i,
        /\bcreate\s+(a\s+)?new\b/i,
        /\bimplement\s+(a\s+)?new\b/i,
        /\bbuild\s+(a\s+)?new\b/i,
        // general "new <noun>" pattern (filtered by relatedToRequirements later)
        /\bnew\s+[a-z]/i,
    ];
    const seemsNewFeature = potentialOutOfScopeRegex.some((re) => re.test(requestLower));
    if (seemsNewFeature) {
        // Check if it's related to any defined feature
        // Build a normalized keyword list from requirements
        // - ignore generic stopwords
        // - include 3+ char tokens (allow 'jwt')
        // - handle simple plural/singular by stripping trailing 's'
        const STOPWORDS = new Set([
            "the",
            "and",
            "with",
            "for",
            "from",
            "into",
            "about",
            "feature",
            "features",
            "system",
            "application",
            "app",
            "service",
            "module",
            "component",
            "screen",
            "page",
            "user",
            "users",
            "management",
            "flow",
            "process",
        ]);
        const escapeRegExp = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        const relatedToRequirements = allFeatures.some((feature) => {
            const rawTokens = feature
                .split(/[^a-z0-9]+/i)
                .map((t) => t.trim())
                .filter((t) => t.length >= 3 && !STOPWORDS.has(t));
            const normalizedTokens = new Set();
            for (const token of rawTokens) {
                normalizedTokens.add(token);
                if (token.endsWith("s") && token.length > 3) {
                    normalizedTokens.add(token.slice(0, -1));
                }
            }
            // Match on word boundaries to avoid substring false-positives
            return Array.from(normalizedTokens).some((token) => {
                const re = new RegExp(`\\b${escapeRegExp(token)}\\b`, "i");
                return re.test(requestLower);
            });
        });
        if (!relatedToRequirements) {
            return {
                inScope: false,
                reason: "This request appears to add functionality not defined in the challenge requirements",
                suggestion: "Please verify this aligns with the requirements in knowledge.yaml. If this is a new requirement, update the knowledge base first.",
            };
        }
    }
    return { inScope: true };
}
