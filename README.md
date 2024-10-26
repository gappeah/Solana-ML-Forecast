# Predicting Solana Price Forecasting with LSTM and SARIMA

This project predicts the future price of Solana using historical data using a hybrid approach of combining  LSTM (Long Short-Term Memory) neural networks, a type of recurrent neural network well-suited for time-series forecasting and SARIMA or Seasonal ARIMA, (Seasonal Autoregressive Integrated Moving Average) is an extension of ARIMA that explicitly supports univariate time series data with a seasonal component. It adds three new hyperparameters to specify the autoregression (AR), differencing (I) and moving average (MA) for the seasonal component of the series, as well as an additional parameter for the period of the seasonality. to model and predict Ethereum prices.


## Project Overview
The primary objective of this project is to predict Ethereum prices using historical data. The model is trained using an LSTM network and the data consists of Ethereum prices from August 2015 to September 2024. The predictions are evaluated using metrics such as **Mean Squared Error (MSE)**, **R-squared (R²)**, **Root Mean Squared Error (RMSE)**, **Mean Absolute Error (MAE)** and **Accuracy** (though accuracy is unconventional in regression tasks, it is also computed here).

### Key Features
- Ensemble Prediction Model
- LSTM (Long Short-Term Memory) neural networks
- SARIMA (Seasonal ARIMA) statistical modeling
- Optimised weight combination for ensemble predictions

- Advanced Technical Indicators
- Simple Moving Averages (7-day and 30-day)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Volatility metrics
- Price momentum indicators

- Market Sentiment Analysis
- Bitcoin correlation tracking
- Market trend indicators
- Volume analysis

## What is Solana 
![5cc0b99a8dd84fbfa4e150d84b5531f2](https://github.com/user-attachments/assets/cb231575-9409-4692-895e-ec7ad6fde6c7)
* Solana is a decentralized computer network that uses a blockchain database to record transactions and manage the currency. The individual unit of Solana is called a sol.
* Solana uses a proof-of-stake (PoS) mechanism and a proof-of-history (PoH) mechanism to improve on the traditional PoS blockchain. PoH uses hashed timestamps to verify when transactions occur.
* Solana can power smart contracts, decentralized finance apps, NFTs, and more. It claims to be able to process 50,000 transactions per second.
* Solana was created by Anatoly Yakovenko and Raj Gokal founded Solana Labs in 2018 and launched Solana in 2020.

### Blockchain Architecture
Solana's blockchain architecture is designed to achieve high performance, scalability, and low transaction costs. Here are the key components and innovations that make up Solana's unique architecture:

## Consensus Mechanism

Solana employs a hybrid consensus mechanism combining Proof of Stake (PoS) and Proof of History (PoH):

**Proof of Stake (PoS)**: Validators are chosen to produce blocks based on the amount of SOL tokens they have staked. This provides security and decentralization to the network[1][4].

**Proof of History (PoH)**: A unique innovation by Solana that acts as a cryptographic clock for the blockchain. PoH creates a historical record that proves an event occurred at a specific moment in time, allowing for efficient ordering of transactions without relying on timestamps[2][4].

## Key Architectural Components

1. **Tower BFT**: An optimized version of Practical Byzantine Fault Tolerance (PBFT) that leverages the PoH clock to reduce communication overhead and latency[2].

2. **Turbine**: A block propagation protocol that breaks data into smaller packets for efficient transmission across the network, increasing bandwidth and transaction capacity[2].

3. **Gulf Stream**: Eliminates the need for a mempool by forwarding transactions to validators before the current block is finalized, reducing confirmation times[2].

4. **Sealevel**: Enables parallel execution of smart contracts, allowing thousands of contracts to run simultaneously without impacting network performance[2].

5. **Pipelining**: A transaction processing unit that optimizes validation times by assigning different stages of transaction processing to specific hardware[2].

6. **Cloudbreak**: Solana's account database, optimized for concurrent reads and writes across the network to achieve necessary scalability[2].

7. **Archivers**: A network of nodes responsible for data storage, offloading this task from validators to maintain efficiency[2].

## Transaction Processing

Solana's Transaction Processing Unit (TPU) follows a four-phase sequence[1]:

1. Data Fetching
2. Signature Verification
3. Banking
4. Writing

This parallelized approach allows Solana to handle up to 50,000 transactions simultaneously[1].

## Network Structure

- **Validators**: The backbone of the network, responsible for processing transactions and maintaining consensus[3].
- **Clusters**: Collections of validators that work together to achieve consensus[3].
- **Slot Leaders**: Validators selected to produce blocks during specific time slots[4].

By combining these innovative features, Solana aims to address the blockchain trilemma of scalability, security, and decentralization, with a particular focus on scalability. This architecture enables Solana to achieve high transaction throughput (up to thousands of transactions per second), low fees, and fast confirmation times (approximately 400ms per block).


**Example: Interacting with Ethereum Blockchain to Get the Latest Block**

```python
from solana.rpc.async_api import AsyncClient
import asyncio

async def get_latest_block():
    # Initialize the Solana client
    client = AsyncClient("https://api.mainnet-beta.solana.com")

    try:
        # Get the latest block
        response = await client.get_latest_blockhash()
        
        if response.value:
            block_hash = response.value.blockhash
            last_valid_block_height = response.value.last_valid_block_height
            
            print(f"Latest block hash: {block_hash}")
            print(f"Last valid block height: {last_valid_block_height}")
            
            # Get more details about the block
            block_details = await client.get_block(last_valid_block_height)
            
            if block_details:
                print(f"Block time: {block_details.block_time}")
                print(f"Number of transactions: {len(block_details.transactions)}")
        else:
            print("Failed to retrieve the latest block information")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    finally:
        # Close the client connection
        await client.close()

# Run the async function
asyncio.run(get_latest_block())
```

---

### Smart Contracts
Smart contracts on Solana, known as "programs," are self-executing pieces of code deployed on the Solana blockchain. Unlike traditional EVM-based contracts, Solana programs are stateless and contain only the logic, while the state is stored in separate accounts. They run on Solana's high-performance runtime, which allows for parallel execution of thousands of contracts simultaneously. Solana programs are primarily written in Rust and compiled to BPF bytecode, leveraging the blockchain's unique features such as Proof of History (PoH) and the account model. This architecture enables Solana smart contracts to achieve exceptional speed, scalability, and cost-efficiency, processing up to 50,000 transactions per second with minimal fees. By separating logic from state and utilizing Solana's innovative consensus mechanism, these programs facilitate rapid, secure, and automated execution of decentralized applications across various domains, including DeFi, NFTs, and gaming, without the need for intermediaries.


**Example: Simple Solana Smart Contract**

```rust
use solana_program::{
    account_info::AccountInfo,
    entrypoint,
    entrypoint::ProgramResult,
    msg,
    pubkey::Pubkey,
};

entrypoint!(process_instruction);

pub fn process_instruction(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    _instruction_data: &[u8],
) -> ProgramResult {
    msg!("Hello, World!");
    
    let account = &accounts[0];
    let mut data = account.try_borrow_mut_data()?;
    let mut num_calls = u32::from_le_bytes(data[0..4].try_into().unwrap());
    num_calls += 1;
    data[0..4].copy_from_slice(&num_calls.to_le_bytes());
    
    msg!("This program has been called {} times", num_calls);
    
    Ok(())
}

```

```rust
use solana_program::{
    account_info::AccountInfo,
    entrypoint,
    entrypoint::ProgramResult,
    pubkey::Pubkey,
    msg,
};
use borsh::{BorshDeserialize, BorshSerialize};

#[derive(BorshSerialize, BorshDeserialize, Debug)]
pub struct CounterAccount {
    pub counter: u32,
}

entrypoint!(process_instruction);

pub fn process_instruction(
    _program_id: &Pubkey,
    accounts: &[AccountInfo],
    instruction_data: &[u8],
) -> ProgramResult {
    let account = &accounts[0];
    let mut counter_account = CounterAccount::try_from_slice(&account.data.borrow())?;

    match instruction_data[0] {
        0 => {
            counter_account.counter += 1;
            msg!("Incremented counter to {}", counter_account.counter);
        },
        1 => {
            counter_account.counter -= 1;
            msg!("Decremented counter to {}", counter_account.counter);
        },
        _ => {
            msg!("Invalid instruction");
            return Err(solana_program::program_error::ProgramError::InvalidInstructionData.into());
        }
    }

    counter_account.serialize(&mut &mut account.data.borrow_mut()[..])?;
    Ok(())
}
```

---


### Consensus Mechanism
Solana uses a hybrid consensus mechanism that combines Proof of Stake (PoS) with a unique innovation called Proof of History (PoH). Here are the key aspects of Solana's consensus mechanism:

Proof of Stake (PoS):
- Validators are chosen to produce blocks based on the amount of SOL tokens they have staked.
- Token holders can delegate their SOL to validators, increasing the validator's stake and voting power.
- This provides security and decentralization to the network.

Proof of History (PoH):
- PoH is a novel timekeeping method for the blockchain.
It creates a historical record that cryptographically proves that an event occurred at a specific moment in time.
- PoH allows for efficient ordering of transactions without relying on timestamps.

Tower BFT:
- An optimized version of Practical Byzantine Fault Tolerance (PBFT).
It leverages the PoH clock to reduce communication overhead and latency in reaching consensus.

Gulf Stream:
- This component forwards transactions to validators before the current block is finalized.
It eliminates the need for a mempool, reducing confirmation times.

Turbine:
- A block propagation protocol that breaks data into smaller packets for efficient transmission across the network.
- This increases bandwidth and transaction capacity.

By combining these elements, Solana's consensus mechanism aims to achieve high throughput (up to thousands of transactions per second), low latency, and fast finality. The PoH serves as a cryptographic clock, allowing the network to agree on the order of events without extensive communication between nodes, while PoS provides the security and incentive structure for validators.

### Use Cases
Ethereum supports various applications across multiple domains:
- **Decentralized Finance (DeFi)**: Platforms built on Ethereum allow users to lend, borrow, trade, and earn interest without traditional financial intermediaries.
- **Non-Fungible Tokens (NFTs)**: Ethereum provides a framework for creating unique digital assets that represent ownership of specific items or content.
- **Decentralized Autonomous Organizations (DAOs)**: These entities operate through smart contracts, allowing members to govern collectively without centralized control.

## Dataset
The dataset used in this project contains Ethereum prices from 2015 to 2024. It is stored in a CSV file (`solana_2020-04-09_2024-10-26.csv`), which includes date-wise prices and other relevant features. The data is preprocessed, normalised, and split into training and test sets before being fed into the LSTM model.

## Setup Instructions

### Prerequisites

Ensure you have Python installed (preferably version 3.7 or higher). You'll also need the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow`
- `keras`
- `pydotplus`
- `graphviz`
- `pydot`
- `kerastuner`

To install the required packages, run the following command:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras kerastuner pydot pydotplus graphviz
```

### Files

- `main_lstm.ipynb`: Jupyter notebook containing the main code for data preprocessing, model training, and evaluation.
- `ethereum_2015-08-07_2024-09-08.csv`: Historical Ethereum price data used to train and test the model.


## Running the Project

1. **Clone the repository** or download the necessary files.
   ```bash
   git clone https://github.com/gappeah/https://github.com/gappeah/Ethereum-Prediction-ML
   ```

2. **Prepare the dataset**: Ensure the `solana_2020-04-09_2024-10-26.csv` file is in the working directory.

3. **Run the Jupyter notebook**: Open the `main_lstm copy.ipynb` file in Jupyter Notebook. The notebook is divided into several steps:
    - Data loading and preprocessing.
    - Splitting the data into training and testing sets.
    - LSTM model creation and training.
    - Model evaluation and visualisation of results.

    You can run the notebook cell-by-cell to execute the entire workflow.

### Code Example

Here's an example of how the LSTM model is defined and trained:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=64)
```

Once trained, you can predict Ethereum prices and evaluate the model using the following metrics:

```python
from sklearn.metrics import mean_squared_error, r2_score

# Make predictions on the test set
y_pred = model.predict(X_test)

# Flatten predictions
y_pred = y_pred.flatten()
y_test = y_test.flatten()

# Calculate MSE and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
```

### Visualising Results

You can visualise the actual vs. predicted Ethereum prices using the following code:

![Ethereum Price Prediction](https://github.com/user-attachments/assets/f8e69e0a-38be-49ec-b875-20329eba58a5)
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 5))
plt.plot(actual_prices, color='blue', label='Actual Ethereum Prices')
plt.plot(predicted_prices, color='red', label='Predicted Ethereum Prices')
plt.title('Ethereum Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
```
![Ethereum Price Prediction After Hyperparameter Tuning](https://github.com/user-attachments/assets/5327f554-6fd6-4240-b232-a4bc176c2304)
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 5))
plt.plot(actual_prices, color='blue', label='Actual Ethereum Prices')
plt.plot(predicted_prices, color='red', label='Predicted Ethereum Prices')
plt.title('Ethereum Price Prediction After Hyperparameter Tuning')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

```
![Model Loss](https://github.com/user-attachments/assets/257e2563-c503-4345-807e-9333471c1d9a)
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 5))
plt.plot(actual_prices, color='blue', label='Actual Ethereum Prices')
plt.plot(predicted_prices, color='red', label='Predicted Ethereum Prices')
plt.title('Ethereum Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
```
![model_architecture](https://github.com/user-attachments/assets/ed4d9ee9-c31e-4a1b-9c7d-6799d6363785)
```python
import matplotlib.pyplot as plt
# Plot training & validation loss values
plt.figure(figsize=(14, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

```

## Model Performance
The model's performance is evaluated using metrics such as:
- **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted prices.
- **R-squared (R²)**: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.
- **Mean Absolute Error (MAE)**: Measures the average absolute difference between actual and predicted prices.
- **Root Mean Squared Error (RMSE)**: Measures the square root of the average squared difference between actual and predicted prices.
- **Accuracy**: For reference, the percentage of predictions that fall within a certain error range from the actual price.

### Example Output:

```
Mean Squared Error Percentage: 0.05%
Root Mean Squared Error (RMSE): 2.54%
Mean Absolute Error (MAE): 1.80%
R-squared Percentage: 98.09%
Accuracy: 98.31288343558282%

```

## Note
 Trend Alignment
Actual vs. Predicted Trends: The blue line (actual Solana prices) should be closely followed by the ensemble prediction (green line) if the model performed well.
Ensemble Model Performance: The ensemble line, combining LSTM and SARIMA, should ideally lie closer to the actual values than either the individual LSTM (purple) or SARIMA (red) predictions.
2. RMSE Confidence Interval
Confidence Band (Green Shaded Area): The green shaded area around the actual values represents the Root Mean Squared Error (RMSE) confidence interval. If your predictions are within this band, it indicates good predictive accuracy.
Deviations: Significant deviations from this confidence band may suggest underfitting (model is too simplistic) or overfitting (model is too complex and lacks generalization).
3. Individual Model Behaviors
LSTM vs. SARIMA:
LSTM (Purple Line): This model might capture non-linear, complex patterns in the data. It could show more variability and adaptability if Solana’s price has rapid fluctuations.
SARIMA (Red Line): SARIMA often captures seasonal and trend components but may struggle with highly volatile changes. It could be smoother or less reactive to sudden spikes compared to LSTM.
Performance Comparison: By examining how closely each model follows the actual trend individually, you can assess the strengths of each. Generally, LSTM is better for capturing volatile patterns, while SARIMA is good for consistent seasonal patterns.
4. Ensemble Model Advantages
Weighted Combination: If your ensemble model outperforms both individual models, you’ll see it tracking the actual trend better than the purple and red lines. This suggests that combining the strengths of both models has led to a more robust prediction.
5. RMSE Score Interpretation
The RMSE score displayed on the plot provides a measure of average prediction error. A lower RMSE value suggests higher prediction accuracy. If the ensemble RMSE is lower than the RMSEs for the individual models, it confirms that combining the models has improved prediction performance.

## References

- [Scikit-learn documentation](https://scikit-learn.org/stable/)
- [TensorFlow documentation](https://www.tensorflow.org/)
- [Keras documentation](https://keras.io/)
- [Graphviz download](https://graphviz.gitlab.io/download/)
- [PyDot download](https://pypi.org/project/pydot/)
- [Pydotplus download](https://pypi.org/project/pydotplus/)
- [Matplotlib documentation](https://matplotlib.org/)
- [Pandas documentation](https://pandas.pydata.org/)
- [Numpy documentation](https://numpy.org/doc/stable/)
