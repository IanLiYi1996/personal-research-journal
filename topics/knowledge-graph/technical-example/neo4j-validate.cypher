// Not executed. These queries detect violations; merely running them does not
// block other writers. Enforce the checks in a controlled write service.

// Required property in the Community-edition application-validation route.
MATCH (r:Release)
WHERE r.version IS NULL OR trim(r.version)=''
RETURN r.uri AS invalidRelease;

// Exactly one correctly typed release target per claim.
MATCH (c:SupportClaim)
OPTIONAL MATCH (c)-[link:FOR_RELEASE]->(r:Release)
WITH c, count(link) AS targets
WHERE targets <> 1
RETURN c.uri AS invalidClaim, targets;

// Each claim must have at least one evidence node.
MATCH (c:SupportClaim)
OPTIONAL MATCH (c)-[link:SUPPORTED_BY]->(e:Evidence)
WITH c, count(link) AS evidenceCount
WHERE evidenceCount=0
RETURN c.uri AS missingEvidence;

// Links must not silently cross model versions.
MATCH (c:SupportClaim)-[:FOR_RELEASE]->(r:Release)
WHERE c.modelVersion IS NULL OR r.modelVersion IS NULL
   OR c.modelVersion <> r.modelVersion
RETURN c.uri AS claim, r.uri AS release;

// State/environment enum checks belong to the controlled write contract.
MATCH (c:SupportClaim)
WHERE c.status IS NULL OR NOT c.status IN ['candidate','approved','revoked']
   OR c.environment IS NULL
   OR NOT c.environment IN ['linux-selfhosted','staging']
RETURN c.uri AS invalidStateOrEnvironment;

// Timestamp types must already have been parsed/validated on input.
MATCH (c:SupportClaim)
WHERE c.validFrom IS NULL
   OR (c.validTo IS NOT NULL AND c.validTo <= c.validFrom)
RETURN c.uri AS invalidInterval;
