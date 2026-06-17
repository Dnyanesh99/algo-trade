# Validation Tests

Comprehensive test suite for the validation system covering all real-world edge cases and different instrument types.

## Test Structure

### 📁 Test Files

- **`test_fixtures.py`** - Test data generators and pytest fixtures
- **`test_candle_validator.py`** - Complete CandleValidator test suite  
- **`test_outlier_detector.py`** - Complete OutlierDetector test suite
- **`test_comprehensive_edge_cases.py`** - Integration tests demonstrating all edge cases
- **`README.md`** - This documentation

### 🧪 Test Coverage

#### CandleValidator Tests (`test_candle_validator.py`)
- ✅ **Initialization & Configuration**: Default/custom configs, error handling
- ✅ **Index Data Validation**: Zero volume, empty OI handling
- ✅ **Equity Data Validation**: Positive volume requirements, filtering
- ✅ **Futures Data Validation**: Volume + OI validation
- ✅ **OHLC Validation**: Conditional validation based on required columns
- ✅ **Outlier Detection**: Configurable column processing
- ✅ **Timestamp Handling**: Column detection, ordering, sorting
- ✅ **Time Gap Detection**: Market break detection
- ✅ **Duplicate Handling**: Timestamp deduplication
- ✅ **NaN Handling**: Required column cleaning
- ✅ **Instrument Types**: INDEX, EQUITY, FNO behavior
- ✅ **Quality Scoring**: Multi-factor quality assessment
- ✅ **Error Handling**: Exception scenarios, malformed data
- ✅ **Configuration Variations**: Custom column combinations
- ✅ **Performance**: Large datasets, real CSV data

#### OutlierDetector Tests (`test_outlier_detector.py`)
- ✅ **IQR Algorithm**: Basic detection, calculation correctness
- ✅ **Handling Strategies**: Clip, remove, flag strategies
- ✅ **Multi-Column Detection**: OHLCV simultaneous processing
- ✅ **Edge Cases**: Empty data, single row, all same values, NaN handling
- ✅ **Column Types**: Datetime skipping, missing columns, strings
- ✅ **Configuration**: IQR multipliers, strategy consistency
- ✅ **Performance**: Large datasets, many columns
- ✅ **Real Data Integration**: CSV data, different instrument types

#### Comprehensive Edge Cases (`test_comprehensive_edge_cases.py`)
- ✅ **Missing Required Columns**: ts, open, high, low, close, volume, oi
- ✅ **Real-World Scenarios**: CSV data, mixed instruments, extreme conditions
- ✅ **Data Corruption**: Negative values, NaN, violations, duplicates, gaps
- ✅ **Configuration Edge Cases**: Minimal configs, custom combinations
- ✅ **Performance & Scalability**: Large datasets, high outlier density

## 🚀 Running Tests

### Prerequisites
```bash
# Navigate to project root
cd /home/dnyanesh/ai-trading/algo-trade

# Ensure dependencies are installed
pip install pytest pandas numpy
```

### Run All Validation Tests
```bash
# Run all validation tests
pytest tests/validation/ -v

# Run with coverage
pytest tests/validation/ --cov=src/validation --cov-report=html

# Run specific test file
pytest tests/validation/test_candle_validator.py -v
pytest tests/validation/test_outlier_detector.py -v
```

### Run Comprehensive Demo
```bash
# Run the comprehensive edge case demonstration
python tests/validation/test_comprehensive_edge_cases.py
```

## 📊 Test Data Sources

### Real CSV Data
- **Source**: `/src/validation/ohlcv_1min_202507191130.csv`
- **Content**: 1,531 rows of real 1-minute NIFTY data
- **Characteristics**: Zero volume (index), empty OI, reverse chronological order

### Simulated Data
- **Index Data**: Zero volume, no OI (like NIFTY)
- **Equity Data**: Positive volume (1k-50k), no OI
- **Futures Data**: Volume (500-10k) + OI (1k-100k)

### Edge Case Data
- **OHLC Violations**: Intentional high < low, high < open/close
- **Statistical Outliers**: 5-10 standard deviations from mean
- **Time Gaps**: 2-hour and 30-minute gaps
- **Duplicates**: Duplicate timestamps
- **NaN Values**: Missing values in critical columns
- **Negative Values**: Invalid negative prices/volumes

## 🎯 Key Test Scenarios

### Configuration-Based Testing
```python
# Index configuration (no volume/OI validation)
required_columns: ["ts", "open", "high", "low", "close"]

# Equity configuration (volume validation)  
required_columns: ["ts", "open", "high", "low", "close", "volume"]

# Futures configuration (volume + OI validation)
required_columns: ["ts", "open", "high", "low", "close", "volume", "oi"]

# Minimal configuration (timestamp + close only)
required_columns: ["ts", "close"]
```

### Instrument Type Testing
```python
# INDEX: Zero volume allowed, no OI
validator.validate(df, instrument_type="INDICES")

# EQUITY: Positive volume required, no OI  
validator.validate(df, instrument_type="EQ")

# FUTURES: Positive volume + valid OI
validator.validate(df, instrument_type="NFO-FUT")
```

### Outlier Strategy Testing
```python
# Clip outliers to bounds
config.outlier_detection.handling_strategy = "clip"

# Remove rows with outliers
config.outlier_detection.handling_strategy = "remove"  

# Flag outliers with boolean columns
config.outlier_detection.handling_strategy = "flag"
```

## 📈 Expected Test Results

### Passing Scenarios
- ✅ **Index data** with zero volume validates successfully
- ✅ **Equity data** with positive volume passes validation
- ✅ **Futures data** with volume + OI validates correctly
- ✅ **Missing columns** are handled gracefully (skip validation)
- ✅ **Real CSV data** processes without errors
- ✅ **Large datasets** (10k+ rows) complete within 30 seconds

### Handled Edge Cases
- ✅ **OHLC violations** are corrected automatically
- ✅ **Outliers** are detected and handled per strategy
- ✅ **Time gaps** are identified and reported
- ✅ **Duplicates** are removed (keeping last)
- ✅ **NaN values** are filtered from required columns
- ✅ **Negative values** are removed from price/volume columns

### Error Scenarios
- ❌ **Empty required_columns** raises ConfigurationError
- ❌ **Invalid quality threshold** (>100) raises ConfigurationError
- ❌ **Missing timestamp** column causes validation failure
- ❌ **All corrupted data** results in empty cleaned dataset

## 🔧 Customizing Tests

### Adding New Test Data
```python
# In test_fixtures.py
@staticmethod
def create_custom_data(rows: int = 100) -> pd.DataFrame:
    base_df = ValidationTestFixtures.load_real_csv_data().head(rows)
    # Add custom modifications
    return base_df
```

### Adding New Test Cases
```python
# In test_candle_validator.py
def test_custom_scenario(self, config_equity):
    validator = CandleValidator(config=config_equity)
    df = ValidationTestFixtures.create_custom_data()
    
    is_valid, cleaned_df, report = validator.validate(
        df, symbol="CUSTOM", instrument_type="EQ"
    )
    
    # Add assertions
    assert is_valid
    assert len(cleaned_df) > 0
```

## 🐛 Debugging Test Failures

### Common Issues
1. **Import Errors**: Ensure `PYTHONPATH` includes project root
2. **Config Errors**: Check `config/config.yaml` is valid
3. **Data Errors**: Verify CSV file exists and is readable
4. **Memory Errors**: Reduce dataset sizes for resource-constrained environments

### Debug Commands
```bash
# Run with verbose output
pytest tests/validation/ -v -s

# Run single test with debug
pytest tests/validation/test_candle_validator.py::TestIndexDataValidation::test_index_data_validation_success -v -s

# Print validation reports
python -c "
from tests.validation.test_fixtures import ValidationTestFixtures
from src.validation.candle_validator import CandleValidator
df = ValidationTestFixtures.load_real_csv_data()
validator = CandleValidator()
_, _, report = validator.validate(df, symbol='DEBUG', instrument_type='INDICES')
print(f'Quality Score: {report.quality_score}')
print(f'Issues: {report.issues}')
"
```

## 📝 Contributing

When adding new tests:
1. Follow existing naming conventions (`test_<functionality>_<scenario>`)
2. Use appropriate fixtures from `test_fixtures.py`
3. Add comprehensive docstrings explaining the test purpose
4. Test both positive and negative scenarios
5. Include performance considerations for large datasets
6. Document any new edge cases discovered

## 🎉 Test Results Summary

The comprehensive test suite validates that the refactored validation system:

✅ **Configurable**: Works with any combination of required columns  
✅ **Robust**: Handles all real-world edge cases gracefully  
✅ **Performant**: Processes large datasets efficiently  
✅ **Accurate**: Correctly identifies and handles data quality issues  
✅ **Flexible**: Supports INDEX, EQUITY, and FUTURES instruments  
✅ **Production-Ready**: Comprehensive error handling and logging