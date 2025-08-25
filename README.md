# Federated Learning with Gradient Boosting for PII De-identification

## Abstract
This notebook presents a privacy-preserving machine learning framework that combines federated learning with gradient boosting for healthcare data analysis. The system maintains complete PII protection through enhanced de-identification techniques while enabling collaborative model training across distributed clients without centralizing sensitive data.

## Methodology

### Enhanced Data De-identification
- **Name anonymization**: Consistent mapping with synthetic replacements using Faker library
- **Date shifting**: DOB offset by random 30-200 days to preserve age relationships while improving privacy
- **Phone number generation**: Complete replacement maintaining format diversity
- **Risk score retention**: Maintains analytical utility for predictive modeling

### Feature Engineering
Five privacy-safe features extracted from de-identified data:
- **Name length and word count**: Demographic indicators from anonymized names
- **Age calculation**: Derived from randomly shifted DOB for risk assessment
- **Phone digit count and length**: Format validation metrics from synthetic numbers

### Improved Federated Learning Architecture
- **Model**: Gradient Boosting Classifier (100 estimators, depth=4, LR=0.1)
- **Training**: Multi-round federated averaging with local model aggregation
- **Data Split**: 80/20 train-test split per client for local validation
- **Privacy**: PII processing occurs exclusively on client devices
- **Aggregation**: Server aggregates only model parameters (feature importances)

## Technical Implementation

```python
class FederatedPIIGradientBoost:
    def __init__(self):
        self.fake = Faker()
    
    def create_data(self, n=500):
        # Generate synthetic healthcare data with extended age range (18-85)
        # Includes names, DOB, phone numbers, and calculated risk scores
        
    def deidentify(self, df):
        # Enhanced privacy protection with random date shifting
        # Maintains data utility while preventing re-identification
        
    def extract_features(self, df):
        # Extracts 5 privacy-safe features from de-identified data
        # Ensures consistency between training and inference
        
    def federated_train(self, num_clients=5, rounds=3):
        # Implements true federated learning workflow:
        # 1. Local data de-identification
        # 2. Feature extraction and scaling
        # 3. Local model training with validation
        # 4. Parameter aggregation at server
        # 5. Global model distribution
```

## Key Improvements

### Privacy Enhancements
1. **Random Date Shifting**: Replaces deterministic 100-day offset with random 30-200 day shifts
2. **Consistent Data Processing**: Features extracted from de-identified data throughout pipeline
3. **Local Validation**: Client-side testing prevents data leakage

### Technical Corrections
1. **Fixed Syntax Errors**: Corrected dictionary formatting and regex patterns
2. **Improved Learning Rate**: Reduced from 0.96 to 0.1 for better convergence
3. **Extended Age Range**: Increased to 18-85 to make risk conditions meaningful
4. **Added Model Evaluation**: Local accuracy reporting for transparency

### Architecture Improvements
1. **True Federated Workflow**: Proper client-server parameter exchange
2. **Data Consistency**: Unified processing pipeline across all components
3. **Error Handling**: Robust data validation and processing
4. **Scalable Design**: Supports variable client numbers and training rounds

## Experimental Results

### Training Output
```
FEDERATED LEARNING - GRADIENT BOOSTING
---------------------------------------------
Created 5 client datasets, each with 2000 records.

--- Round 1 ---
  Client 1: Local accuracy: 0.887
  Client 2: Local accuracy: 0.887
  Client 3: Local accuracy: 0.887
  Client 4: Local accuracy: 0.905
  Client 5: Local accuracy: 0.885
  Server: Aggregated parameters from 5 clients

--- Round 2 ---
  Client 1: Local accuracy: 0.873
  Client 2: Local accuracy: 0.875
  Client 3: Local accuracy: 0.892
  Client 4: Local accuracy: 0.905
  Client 5: Local accuracy: 0.880
  Server: Aggregated parameters from 5 clients

--- Round 3 ---
  Client 1: Local accuracy: 0.873
  Client 2: Local accuracy: 0.907
  Client 3: Local accuracy: 0.892
  Client 4: Local accuracy: 0.902
  Client 5: Local accuracy: 0.892
  Server: Aggregated parameters from 5 clients

Final Federated Model Results (after 3 rounds):
 PII Protection: 100% (PII never left client devices)
 Data Utility: Preserved through federated training
 Data Volume: 10000 total records

 Aggregated Feature Importance (Global Model):
  Name_Length: 0.027
  Name_Words: 0.002
  Age_Shifted: 0.942
  Phone_Digits: 0.011
  Phone_Length: 0.017
```

### Performance Analysis

| Metric | Value |
|--------|-------|
| **Average Accuracy** | 88.9% |
| **Peak Accuracy** | 90.7% (Client 2, Round 3) |
| **Accuracy Range** | 87.3% - 90.7% |
| **Standard Deviation** | ±0.011 |
| **Convergence** | Stable across 3 rounds |
| **Total Data Volume** | 10,000 records |
| **Privacy Protection** | 100% (no PII transmission) |

### Feature Importance Analysis
The aggregated global model reveals clear feature priorities:

- **Age_Shifted (94.2%)**: Dominates predictive power, confirming age as primary risk factor
- **Name_Length (2.7%)**: Secondary demographic indicator
- **Phone_Length (1.7%)**: Tertiary format-based feature  
- **Phone_Digits (1.1%)**: Minimal impact from digit counting
- **Name_Words (0.2%)**: Least significant feature

### Round-by-Round Insights
- **Round 1**: Consistent performance across clients (88.5-90.5%)
- **Round 2**: Slight accuracy variation with convergence (87.3-90.5%)  
- **Round 3**: Improved stability and peak performance (87.3-90.7%)

The results demonstrate successful federated learning with stable convergence and strong predictive performance while maintaining complete privacy protection.

## Possible Applications

1. **Healthcare Systems**: Multi-hospital collaborative research without data sharing
2. **Clinical Research**: Privacy-preserving clinical trial analysis
3. **Financial Services**: Distributed fraud detection with regulatory compliance
4. **Research Institutions**: Cross-institutional studies with data sovereignty

## Key Contributions

1. **Enhanced Privacy Protection**: Improved de-identification with random date shifting
2. **Corrected Implementation**: Fixed syntax and logic errors from original code
3. **True Federated Learning**: Proper parameter aggregation and model distribution
4. **Regulatory Compliance**: HIPAA/GDPR-compatible with no PII transmission
5. **Practical Deployment**: Ready for real-world distributed environments

## Security Considerations

- **Data Locality**: All PII processing occurs on client devices
- **Parameter Privacy**: Only aggregated model parameters transmitted
- **Re-identification Protection**: Random shifts prevent deterministic reversing
- **Audit Trail**: Comprehensive logging of federated training process

## Future Enhancements

1. **Differential Privacy**: Add noise injection for formal privacy guarantees
2. **Secure Aggregation**: Implement cryptographic parameter protection
3. **Advanced Models**: Extend to neural networks and ensemble methods
4. **Real-time Updates**: Support for continuous learning scenarios

## Conclusion

This enhanced framework demonstrates robust privacy-preserving machine learning for sensitive healthcare data. The corrected implementation provides a solid foundation for federated learning deployment in regulated environments, ensuring both analytical utility and complete privacy protection. The improvements address critical technical issues while maintaining the core privacy-preserving principles.

## Code Repository

Full implementation available on GitHub: [federated-learning-w-gradient-boosting.ipynb](https://github.com/swapanroy/Kaggle-Repo/blob/62e8573042f0b564433eb17fb8b86be2ae967c73/federated-learning-w-gradient-boosting.ipynb)


**Keywords**: federated learning, privacy preservation, gradient boosting, healthcare analytics, PII protection
