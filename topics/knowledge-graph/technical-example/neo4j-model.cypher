// Native property-graph route. No RDF import, APOC or n10s is required.
// Documentation example, NOT executed against a Neo4j server.
// Use only an isolated demo database. Fixture MERGE/SET is not a production sync.

CREATE CONSTRAINT ontology_uri IF NOT EXISTS
FOR (v:OntologyVersion) REQUIRE v.uri IS UNIQUE;
CREATE CONSTRAINT entity_type_identity IF NOT EXISTS
FOR (t:EntityType) REQUIRE (t.modelVersion, t.iri) IS UNIQUE;
CREATE CONSTRAINT relation_type_identity IF NOT EXISTS
FOR (t:RelationType) REQUIRE (t.modelVersion, t.iri) IS UNIQUE;
CREATE CONSTRAINT property_identity IF NOT EXISTS
FOR (p:PropertyDefinition) REQUIRE (p.modelVersion, p.iri) IS UNIQUE;
CREATE CONSTRAINT product_uri IF NOT EXISTS
FOR (p:Product) REQUIRE p.uri IS UNIQUE;
CREATE CONSTRAINT release_uri IF NOT EXISTS
FOR (r:Release) REQUIRE r.uri IS UNIQUE;
CREATE CONSTRAINT capability_uri IF NOT EXISTS
FOR (k:Capability) REQUIRE k.uri IS UNIQUE;
CREATE CONSTRAINT claim_uri IF NOT EXISTS
FOR (c:SupportClaim) REQUIRE c.uri IS UNIQUE;
CREATE CONSTRAINT evidence_uri IF NOT EXISTS
FOR (e:Evidence) REQUIRE e.uri IS UNIQUE;

// Definition graph. These nodes document semantics; they do not enforce them.
MERGE (v:OntologyVersion {uri:'urn:onto-tech:model:0.1.0'})
SET v.version='0.1.0', v.status='active'
MERGE (pt:EntityType {modelVersion:'0.1.0', iri:'https://example.org/onto-tech/Product'})
SET pt.name='Product', pt.definition='A logical product, not a release.'
MERGE (rt:EntityType {modelVersion:'0.1.0', iri:'https://example.org/onto-tech/Release'})
SET rt.name='Release', rt.definition='A particular product release.'
MERGE (et:EntityType {modelVersion:'0.1.0', iri:'https://example.org/onto-tech/EnterpriseRelease'})
SET et.name='EnterpriseRelease'
MERGE (ct:EntityType {modelVersion:'0.1.0', iri:'https://example.org/onto-tech/SupportClaim'})
SET ct.name='SupportClaim'
MERGE (kt:EntityType {modelVersion:'0.1.0', iri:'https://example.org/onto-tech/Capability'})
SET kt.name='Capability'
MERGE (ev:EntityType {modelVersion:'0.1.0', iri:'https://example.org/onto-tech/Evidence'})
SET ev.name='Evidence'
MERGE (v)-[:CONTAINS_TYPE]->(pt)
MERGE (v)-[:CONTAINS_TYPE]->(rt)
MERGE (v)-[:CONTAINS_TYPE]->(et)
MERGE (v)-[:CONTAINS_TYPE]->(ct)
MERGE (v)-[:CONTAINS_TYPE]->(kt)
MERGE (v)-[:CONTAINS_TYPE]->(ev)
MERGE (et)-[:SUBCLASS_OF]->(rt)
MERGE (prop:PropertyDefinition {
    modelVersion:'0.1.0', iri:'https://example.org/onto-tech/version'
})
SET prop.name='version', prop.datatype='STRING', prop.required=true
MERGE (rt)-[:HAS_PROPERTY]->(prop)
MERGE (rel:RelationType {
    modelVersion:'0.1.0', iri:'https://example.org/onto-tech/forRelease'
})
SET rel.name='FOR_RELEASE', rel.minTargets=1, rel.maxTargets=1
MERGE (v)-[:CONTAINS_RELATION]->(rel)
MERGE (rel)-[:FROM_TYPE]->(ct)
MERGE (rel)-[:TO_TYPE]->(rt)
RETURN v.version AS modelVersion;

// Instance graph. Parent labels are deliberately materialized by the importer.
MATCH (pt:EntityType {modelVersion:'0.1.0', name:'Product'}),
      (rt:EntityType {modelVersion:'0.1.0', name:'Release'}),
      (et:EntityType {modelVersion:'0.1.0', name:'EnterpriseRelease'}),
      (ct:EntityType {modelVersion:'0.1.0', name:'SupportClaim'}),
      (kt:EntityType {modelVersion:'0.1.0', name:'Capability'}),
      (ev:EntityType {modelVersion:'0.1.0', name:'Evidence'})
MERGE (p:Product {uri:'https://example.org/onto-tech/ProductA'})
SET p.modelVersion='0.1.0', p.tenantId='tenant-demo'
MERGE (r21:Release {uri:'https://example.org/onto-tech/Release21'})
SET r21:EnterpriseRelease, r21.version='2.1',
    r21.modelVersion='0.1.0', r21.tenantId='tenant-demo'
MERGE (r20:Release {uri:'https://example.org/onto-tech/Release20'})
SET r20.version='2.0', r20.modelVersion='0.1.0', r20.tenantId='tenant-demo'
MERGE (k:Capability {uri:'https://example.org/onto-tech/BatchInference'})
SET k.modelVersion='0.1.0'
MERGE (e21:Evidence {uri:'https://example.org/onto-tech/Evidence21'})
SET e21.quote='虚构资料：A 2.1 在 Linux 自托管环境支持批量推理。',
    e21.locator='demo-doc-v3，部署支持，第2段',
    e21.modelVersion='0.1.0', e21.tenantId='tenant-demo'
MERGE (e20:Evidence {uri:'https://example.org/onto-tech/Evidence20'})
SET e20.quote='虚构资料：A 2.0 的支持声明于2026-10-20结束。',
    e20.locator='demo-doc-v2，第4段',
    e20.modelVersion='0.1.0', e20.tenantId='tenant-demo'
MERGE (c21:SupportClaim {uri:'https://example.org/onto-tech/Claim21'})
SET c21.environment='linux-selfhosted', c21.status='approved',
    c21.validFrom=datetime('2026-10-20T00:00:00+08:00'),
    c21.validTo=null, c21.modelVersion='0.1.0', c21.tenantId='tenant-demo'
MERGE (c20:SupportClaim {uri:'https://example.org/onto-tech/Claim20'})
SET c20.environment='linux-selfhosted', c20.status='approved',
    c20.validFrom=datetime('2026-10-01T00:00:00+08:00'),
    c20.validTo=datetime('2026-10-20T00:00:00+08:00'),
    c20.modelVersion='0.1.0', c20.tenantId='tenant-demo'
MERGE (p)-[:INSTANCE_OF]->(pt)
MERGE (r21)-[:INSTANCE_OF]->(et)
MERGE (r20)-[:INSTANCE_OF]->(rt)
MERGE (k)-[:INSTANCE_OF]->(kt)
MERGE (e21)-[:INSTANCE_OF]->(ev)
MERGE (e20)-[:INSTANCE_OF]->(ev)
MERGE (c21)-[:INSTANCE_OF]->(ct)
MERGE (c20)-[:INSTANCE_OF]->(ct)
MERGE (r21)-[:RELEASE_OF]->(p)
MERGE (r20)-[:RELEASE_OF]->(p)
MERGE (c21)-[:FOR_RELEASE]->(r21)
MERGE (c21)-[:ASSERTS]->(k)
MERGE (c21)-[:SUPPORTED_BY]->(e21)
MERGE (c20)-[:FOR_RELEASE]->(r20)
MERGE (c20)-[:ASSERTS]->(k)
MERGE (c20)-[:SUPPORTED_BY]->(e20)
RETURN c21.uri AS currentClaim, c20.uri AS expiredClaim;
