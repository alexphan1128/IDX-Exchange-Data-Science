# Week 1 — Orientation and Setup

## Project goal

The goal is to predict a California property's final sale price (`ClosePrice`) from characteristics known about the property. The analysis is restricted to residential single-family homes.

## Dataset access

I downloaded seven monthly `CRMLSSold` files from `/raw/California`: June 2024, July 2024, September 2024, October 2024, November 2024, December 2024, and January 2025. This meets the six-month minimum, although August 2024 was not available in the downloaded files. The files were copied locally and were not modified on the server.

## Key columns

| Column | Meaning and use |
|---|---|
| `ClosePrice` | Final sale price; this is the target the model will predict. |
| `CloseDate` | Date the property sale closed; used to create monthly train/test periods. |
| `ListingContractDate` | Date the seller and listing agent signed the listing agreement. |
| `PurchaseContractDate` | Date the buyer and seller signed the purchase agreement, usually before closing. |
| `LivingArea` | Interior living area, generally measured in square feet; likely an important numerical predictor. |
| `BedroomsTotal` | Number of bedrooms. |
| `BathroomsTotalInteger` | Number of bathrooms recorded as an integer. |
| `LotSizeSquareFeet` | Lot size in square feet. |
| `YearBuilt` | Year the property was built; can represent property age. |
| `DaysOnMarket` | Number of days the property was listed before closing. |
| `ListPrice` | Original/listing price; a potentially useful market-price feature. |
| `UnparsedAddress` | Full address as entered by the listing agent; retained for context but not used directly as a model feature. |
| `Latitude`, `Longitude` | Geographic coordinates; useful for regional price differences. |
| `City`, `CountyOrParish` | Categorical geographic fields that can be encoded. |
| `PropertyType` | Broad property category; required to equal `Residential`. |
| `PropertySubType` | More specific category; required to equal `SingleFamilyResidence`. |
| `ListingKey` | Listing identifier; used to remove duplicate listings, not as a model feature. |

## Initial data summary

- Combined rows before filtering: 154,367
- Rows after the required property filter and valid positive target check: 78,580
- Date range: 2024-06-01 through 2025-01-31
- Monthly filtered row counts: {'2024-06': 12537, '2024-07': 13374, '2024-09': 10895, '2024-10': 12340, '2024-11': 10672, '2024-12': 10618, '2025-01': 8144}
- Median `ClosePrice`: $900,000
- Median `LivingArea`: 1,788 square feet

## Important limitation

The files contain a gap in August 2024. The preprocessing notebook uses January 2025 as the test month and the available months immediately before it as candidate training windows.

## Real-estate context

`ListPrice` represents the seller's asking price, while `ClosePrice` represents the final transaction value. The transaction typically moves from listing, to purchase agreement and escrow, to closing. Because these are sold records, `CloseDate` and `ClosePrice` are available as the modeling target and time variable.
