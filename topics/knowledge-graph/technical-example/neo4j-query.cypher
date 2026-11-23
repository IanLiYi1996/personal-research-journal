// Not executed. Bind parameters in the trusted backend, not via string concat.
// tenantId must come from verified auth context. A tenant filter is not a full ACL.
WITH datetime($at) AS at
MATCH (c:SupportClaim {tenantId:$tenantId, modelVersion:$modelVersion})
      -[:FOR_RELEASE]->(r:Release {tenantId:$tenantId, modelVersion:$modelVersion})
      -[:RELEASE_OF]->(p:Product {tenantId:$tenantId, modelVersion:$modelVersion})
MATCH (c)-[:ASSERTS]->(k:Capability {
    uri:$capabilityUri, modelVersion:$modelVersion
})
MATCH (c)-[:SUPPORTED_BY]->(e:Evidence {
    tenantId:$tenantId, modelVersion:$modelVersion
})
WHERE c.environment=$environment
  AND c.status='approved'
  AND c.validFrom <= at
  AND (c.validTo IS NULL OR at < c.validTo)
RETURN DISTINCT c.uri AS claim, p.uri AS product, r.uri AS release,
       r.version AS version, e.uri AS evidence, e.quote AS quote,
       e.locator AS locator;

// Use the versioned definition graph to query class ancestry.
// This is explicit path matching, not arbitrary OWL reasoning.
MATCH (target:EntityType {modelVersion:$modelVersion, iri:$targetType})
MATCH (object)-[:INSTANCE_OF]->(actual:EntityType)
      -[:SUBCLASS_OF*0..]->(target)
WHERE object.modelVersion=$modelVersion
  AND object.tenantId=$tenantId
RETURN DISTINCT object.uri;
