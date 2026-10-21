# Nium Sandbox Verification Spike

**Date**: 2026-10-20  
**Sandbox base URL**: `https://gateway.nium.com/api`  
**Client**: Industry Builder (`clientHashId` from Secrets Manager)  
**Existing clear customer reused**: `4f792004-32c7-424b-81ab-348d0f12cd2e`  
**Wallet**: `8113e10a-8d6c-4978-8c8c-acd9d4286343`

---

## Step 1 — Authentication ✅

**Endpoint tested**: `GET /api/v1/client/{clientHashId}`  
**Result**: HTTP 200  
**Response key fields**: `name: "Industry Builder"`, `markup: 0.0`, `countryCode: "SG"`, `regulatoryRegion: "SG"`, `fundingInstrumentType: "RESTRICTED"`

Auth header `x-api-key` works. Confirmed by receiving business-logic errors (400/404) rather than 401 on all subsequent calls.

**Note**: AWS API Gateway returns 403 `"Missing Authentication Token"` for URL paths that don't match any route — these are not auth failures but missing routes.

---

## Step 2 — Foreign Exchange ✅ (partial)

### 2a. FX Rate Quote

**Endpoint**: `POST /api/v1/client/{clientHashId}/quotes`  
**Working pair**: USD → SGD (without `lockPeriod`)  
**Request**:
```json
{
  "sourceCurrencyCode": "USD",
  "destinationCurrencyCode": "SGD",
  "conversionSchedule": "immediate",
  "quoteType": "balance_transfer"
}
```
**Response** (HTTP 200):
- `id`: `quote_30kv4IAiVHH50VMocNSBc2`
- `exchangeRate`: 1.2784
- `netExchangeRate`: 1.2758432
- `markupRate`: 0.0025568
- `expiryTime`: ~60 min from creation

### 2b. FX Market Rate Conversion (locked-quote-free)

**Endpoint**: `POST /api/v1/client/{clientHashId}/customer/{customerHashId}/wallet/{walletHashId}/conversions`  
**Request**:
```json
{"sourceCurrencyCode": "SGD", "destinationCurrencyCode": "USD", "sourceAmount": 100}
```
**Response** (HTTP 200):
- `id`: `conversion_64VmKJBwxLO6kn6UFiAxzf`
- `status`: `processing`
- `sourceAmount`: 100, `destinationAmount`: 78.03
- `netExchangeRate`: 0.7803362, `exchangeRate`: 0.7819, `markupRate`: 0.0015638
- `systemReferenceNumber`: `WFT9584540977`

Executed twice in the same session — both returned `status: "processing"`. No webhook URL registered so completion event not observed.

### 2c. VND Not Supported ❌

```
HTTP 400: {"code": "fx_invalid_format_input", "description": "VND is not valid currency for client"}
```

The client config confirms: VND is absent from the `currencies` array. Only SGD, USD, GBP, AUD, HKD, EUR are enabled.

### 2d. FX Locked Rate (8-hour lock) ❌

```
HTTP 404: {"code": "resource_not_found", "description": "Fee configuration not found for feeType: FX_MARKUP_LOCK_8HOURS"}
```

The locked-rate quote (`lockPeriod` field) requires `FX_MARKUP_LOCK_8HOURS` fee configuration, which is not set up for this sandbox client. Market-rate conversion works as a workaround.

---

## Step 3 — Customer Onboarding ✅ (pre-existing)

**Finding**: One existing corporate customer in `status: clear, subStatus: null` was found via `GET /api/v5/client/{clientHashId}/customers`. Reused for all subsequent steps.

### Customer Creation Attempts (for new sandbox setups)

**v5 API** `POST /api/v5/client/{clientHashId}/customers`:
- Required fields discovered iteratively: `region` (mandatory, value `"SG"`), `kycType`, `type` (exact enum unknown)
- Errors encountered: `"region is required"` → `"kycType is required"` → `"type is required"`
- Full corporate body requires `businessDetails` (with nested `addresses.registeredAddress`, `stakeholders`, `applicantDetails`) and `riskAssessmentInfo`

**v1 API** `POST /api/v1/client/{clientHashId}/corporate`:
- Requires same `businessDetails` structure with `region: "SG"`
- lastName validation: no numbers allowed  
- All field requirements identical to v5 shape

**Simulation**: `POST /api/v5/simulations/onboard/{customerHashId}/transition` with `{"nextAction": "clear"}`  
**Tip**: Include `M01` in the BRN for sandbox auto-approval.

**Status flow**: `awaiting_kyc` → (submitKyc) → `under_review` → (simulate) → `clear`

---

## Step 4 — Wallet Funding ✅ (partial)

### 4a. BANK_TRANSFER Funding

**Endpoint**: `POST /api/v1/client/{clientHashId}/customer/{customerHashId}/wallet/{walletHashId}/fund`  
**Working fundingChannel**: `BANK_TRANSFER`  
**Request**:
```json
{
  "amount": 1000,
  "fundingChannel": "BANK_TRANSFER",
  "sourceCurrencyCode": "USD",
  "destinationCurrencyCode": "USD"
}
```
**Response** (HTTP 200):
- `systemReferenceNumber`: `FW9629003097`
- `status`: `Pending`
- `paymentMethods`: `[]` (no specific payment method returned)

**Note**: Status is `Pending` — this represents a bank transfer initiation. The wallet balance doesn't credit immediately; it requires the simulate-incoming-payment flow or a real DBS/JPM incoming wire.

Valid `fundingChannel` values: `PREFUND`, `BANK_TRANSFER`, `CARD`, `DIRECT_DEBIT`, `FASTER_DIRECT_DEBIT`

### 4b. PREFUND

```
HTTP 400: "Prefund is low" / "Client prefund balance is not sufficient to prefund wallet"
```

The Developer Portal sandbox prefund balance is depleted.

### 4c. Wallet Balances Observed

The wallet already holds SGD 5000 (pre-existing balance):

| Currency | Balance |
|----------|---------|
| SGD | 5,000.00 |
| USD | 0.00 |
| GBP | 0.00 |
| AUD | 0.00 |
| HKD | 0.00 |
| EUR | 0.00 |

VND pocket does not exist.

---

## Step 5 — Local Payment (VN) ❌

### 5a. Beneficiary Creation — Blocked

**Endpoint**: `POST /api/v2/client/{clientHashId}/customer/{customerHashId}/beneficiaries`  

All attempts returned:
```
HTTP 400: "Invalid beneficiary account type" / "Account Type should be Individual or Corporate"
```

This error fires for EVERY request regardless of:
- `accountType` value ("Individual", "Corporate", "INDIVIDUAL", "CORPORATE", `null`, and ~10 other variants)
- Currency (VND, SGD, USD all tried)
- Country (VN, SG, US all tried)

**Root cause diagnosis**: The client config shows `fundingInstrumentType: "RESTRICTED"` and `cardTxnProductCode: null`. The client's `currencies` array contains no VND. The sandbox program appears configured for payin/FX only, with outbound remittance not fully activated. This is a program-level configuration issue, not an API usage error.

### 5b. Remittance (inline beneficiary)

All attempts:
```
HTTP 400: "beneficiary details should be present."
```

This fires even with full inline beneficiary fields in the remittance body (tried: `beneficiaryName`, `beneficiaryAccountNumber`, `beneficiaryCountryCode`, `beneficiaryDetails` nested object, etc.) — the root cause is the same as 5a: no beneficiary can be created or referenced.

### 5c. VND Configuration

VND is not in the client's `currencies` array. Both the FX and remittance layers confirm this. VND payouts require Nium account team to add VND to the client program.

**Documentation confirms 6 LOCAL payout routes for VN** (from prior research), and the test account numbers are:
- PAID: `000987654321`
- RETURN: `000897654321`

---

## Step 6 — Virtual Card ❌

**Endpoint**: `POST /api/v2/client/{clientHashId}/customer/{customerHashId}/wallet/{walletHashId}/card`

**Discovery path**:
1. `"plasticId cannot be null/empty"` — field is required
2. `"Plastic id must be numeric only"` — only numeric values accepted
3. `"Logo id or logo identifier or region is not found"` — numeric IDs not found in card program config
4. `"Plastic id is not found"` — specific IDs fail lookup

**Client config**: `cardTxnProductCode: null`, `applePaySupport: false`, `googlePaySupport: false`

`plasticId` and `cardProductId` are numeric IDs provisioned by Nium when setting up a card program. They are account-specific and cannot be discovered via API.

### Card Simulation Endpoints

No card auth simulation endpoint was found:
- `POST /api/v1/simulations/cards/{cardHashId}/authorize` → 404
- `POST /api/v2/simulations/cards/{cardHashId}/transaction` → 404

Card simulation is not documented in sandbox testing docs.

---

## Step 7 — Webhooks ⚠️

**Status**: Documented, not testable in sandbox

**Documented webhook events**:
- `FX_CONVERSION_COMPLETED`, `FX_CONVERSION_CANCELLED`, `FX_CONVERSION_FAILED`
- `WALLET_FUNDED`
- Payout events: `PAYOUT_PAID`, `PAYOUT_RETURN`
- Card events: `ADD_CARD`, `ASSIGN_CARD`

**Client config**: `notificationWebhook: null` — no webhook URL registered

**Sandbox delivery simulation**: None found. No `/simulations/webhooks` endpoint exists in docs.

---

## Client Configuration Summary

```json
{
  "name": "Industry Builder",
  "countryCode": "SG",
  "regulatoryRegion": "SG",
  "licenseEntity": "NIUM",
  "fundingInstrumentType": "RESTRICTED",
  "cardTxnProductCode": null,
  "currencies": ["SGD", "USD", "GBP", "AUD", "HKD", "EUR"],
  "postFundedPayout": false,
  "paymentIds": [{
    "currencyCode": "SGD",
    "bankName": "JPM_SG",
    "routingCodeType1": "SWIFT",
    "routingCodeValue1": "CHASSGSG",
    "accountType": "LOCAL+GLOBAL"
  }]
}
```

---

## Status Summary

| Step | Result | Notes |
|------|--------|-------|
| Auth | ✅ | `GET /api/v1/client/{id}` returns 200 |
| FX Quote | ✅ | USD→SGD works; VND not in client currencies |
| FX Conversion | ✅ | SGD→USD market rate, status=`processing` |
| Customer (existing) | ✅ | Pre-existing `clear` customer reused |
| Customer creation | ⚠️ | v5/v1 both work structurally; `M01` in BRN = auto-approve |
| Wallet fund | ⚠️ | BANK_TRANSFER initiates (Pending); PREFUND balance depleted |
| VN Beneficiary | ❌ | `fundingInstrumentType: RESTRICTED`; VND not in currency list |
| VN Remittance | ❌ | Blocked by beneficiary failure |
| Virtual card | ❌ | `cardProductId`/`plasticId` not provisioned; `cardTxnProductCode: null` |
| Card simulation | ❌ | No sandbox endpoint found |
| Webhooks | ⚠️ | Documented in docs; no sandbox delivery sim |

---

## Pits and Gotchas

1. **AWS API Gateway 403 "Missing Authentication Token"** is not an auth error — it means the URL path doesn't match any route.
2. **FX locked rate** requires `FX_MARKUP_LOCK_8HOURS` fee config. Use market-rate conversion instead.
3. **VND not a client currency** — both FX and remittance blocked. Must be added by Nium account team.
4. **`fundingInstrumentType: "RESTRICTED"`** blocks beneficiary creation for all currencies, not just VND. This is the primary payout blocker.
5. **`plasticId` and `cardProductId` are numeric program IDs** not discoverable via API — must be requested from Nium during program setup.
6. **FX conversion status stays `processing`** in sandbox without a webhook; no simulation endpoint to advance it to `completed`.
7. **PREFUND balance is depleted** on the Developer Portal sandbox. Use BANK_TRANSFER for funding.
8. **Customer v5 creation requires** `region`, `kycType`, `type` plus the full `businessDetails` / `riskAssessmentInfo` structure.
9. **`M01` in BRN triggers sandbox auto-approve** — skip the submitKyc + simulate sequence.
10. **The v1 beneficiary endpoint** (`/api/v1/.../beneficiaries`) returns 403 (route not found). Use v2.

---

## Demo Design Implications

### What Can Run Live (Today, With Current Sandbox)

| Track | What Works | Estimated Duration |
|-------|------------|-------------------|
| FX | Quote rate (`/quotes`), execute conversion (`/conversions`) | ~5 sec per call |
| Auth + Client info | Immediate | 1 sec |
| FX conversion flow | Quote → confirm conversion → show rate/result | 15–30 sec |

### What Requires Pre-setup (Cannot Improvise Live)

| Track | Blocker | What Nium Must Do |
|-------|---------|------------------|
| VN Local Payment | VND not in client currencies; `RESTRICTED` fundingInstrumentType | Enable VND currency + unlock payout capability for sandbox program |
| VN Local Payment | Beneficiary creation blocked | Same as above |
| Virtual Card | No card program configured | Provision `plasticId` and `cardProductId` in sandbox |
| FX Locked Rate | `FX_MARKUP_LOCK_8HOURS` not configured | Optional — market rate works fine |
| Webhook demo | No webhook URL configured | Register a webhook endpoint before demo |

### Recommended Pre-demo Checklist

- [ ] Contact Nium Developer Support to enable VND outbound payments and unlock payout for the sandbox program
- [ ] Request `plasticId` and `cardProductId` values for a SG virtual card program
- [ ] Pre-fund sandbox wallet (BANK_TRANSFER simulation or top-up via Nium portal)
- [ ] Register a webhook URL (can be a RequestBin or similar)
- [ ] Pre-create and approve a customer so `customerHashId` and `walletHashId` are ready
- [ ] Confirm VND appears in `GET /api/v1/client/{id}` currencies list before demo day

### Live Demo Feasibility (Per Track)

| Track | Live Demo? | Fallback |
|-------|-----------|----------|
| FX (SGD↔USD) | ✅ Fully live | — |
| VN Local Payment | ❌ Needs pre-setup | Pre-run + show logs |
| Agent Virtual Card | ❌ Needs card program setup | Pre-run + show logs |

### Approx End-to-End Timing (if fully configured)

- FX quote + conversion: 5–10 sec
- Customer onboard to clear: ~30–60 sec (with simulation API)
- Wallet fund → confirmed: ~10 sec (simulate incoming payment)
- Beneficiary create + remittance to PAID: ~60–90 sec (includes polling)
- Virtual card issue: 5 sec
- Full demo path: **~3–5 min** end-to-end
