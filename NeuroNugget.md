# NeuroNugget: TechTots - A Baby AI Development Stack

![TechTots Logo](techtots-logo.png)

## Overview

TechTots is an innovative project designed to introduce natural language processing (NLP) into the world of Baby AI development. This tech stack, known as NeuroNugget, enables the creation of a nurturing environment for a Baby AI, named TechTot. TechTot interacts with its best friend, a Monster Toy, as well as with parents, facilitating the development of early AI-human connections.

TechTot is capable of understanding and responding in the "Huuuki" language, which is a significant part of its development.

NeuroNugget leverages the following technologies:

- Keras for Natural Language Processing (NLP).
- ML Agents for Baby AI learning in a controlled environment.
- FaunaDB for efficient data storage, hosted on the TechTot sidechain.
- Blockchain technology for secure, transparent, and tamper-proof data management.

This README provides an essential guide to NeuroNugget, including its setup, components, usage, and key data interactions.

## Components

### 1. TechTot (Baby AI)

TechTot serves as the central character in the TechTots project. It is an AI entity designed to grow and learn through interactions with its surroundings, primarily focusing on language and emotional development. TechTot communicates in the "Huuuki" language, a unique aspect of its development.

### 2. Monster Toy

The Monster Toy is TechTot's best friend and playmate. It engages in interactive activities, games, and conversations with TechTot to stimulate its cognitive and emotional growth.

### 3. Parental Interaction

TechTot also engages with parents, providing them with insights into their child's development and creating a bond between the AI and the family.

### 4. Keras for NLP (Integrated Summary)

The `KerasModel` script enhances TechTot's interactive capabilities by enabling it to engage in meaningful conversations with users in the "Huuuki" language. Here's a summary of its key operations:

- **User Input Handling**: The script reads user input from a Unity `TMP_InputField`.

- **Tokenization**: User input is tokenized and prepared for the neural network.

- **Model Execution**: The Keras NLP model is executed using a dummy input tensor.

- **Response Generation**: A token index is sampled from the model's output using a softmax-based approach.

- **AI Response**: The sampled token index is converted into a generated AI response based on a predefined vocabulary.

- **Conversation History**: The script maintains a conversation history to keep track of user interactions and AI responses.

- **Displaying AI Response**: The generated AI response is displayed in a Unity `TMP_Text` component.

- **Data Storage**: Conversation data, including user inputs and AI responses, is stored on FaunaDB for analysis and further AI development.

This script significantly enhances the interactive capabilities of TechTot, allowing it to engage in meaningful conversations with users in the "Huuuki" language.

### 5. ML Agents (Summary Integrated)

In Unity ML-Agents, the `mlagents.camera` sensor and the `vis_encode_type` parameter are used to capture and process visual observations from the agent's perspective during reinforcement learning. These components enable the agent to "see" its environment and make decisions based on visual data. The choice of sensor configuration and encoding type depends on the specific requirements and complexity of TechTots reinforcement learning task.

### 6. FaunaDB

FaunaDB is used as the data storage solution for TechTot. It hosts the TechTot sidechain, ensuring secure and efficient storage of all data generated during interactions and learning processes.

### 7. Blockchain Technology

Blockchain technology is seamlessly integrated into TechTots to provide secure, transparent, and tamper-proof data management. The blockchain interactions include:

- **Data Integrity:** All interactions, transactions, and learning progress are recorded on the blockchain, ensuring that the data remains unaltered and verifiable.

- **Proof of Work Mining:** TechTot's movements are mined based on a proof-of-work mechanism, adding a layer of security and transparency to its interactions.

- **Smart Contracts:** Smart contracts are employed to automate certain processes within the TechTot ecosystem, enhancing efficiency and trust.

- **Transparency:** Users, including parents, can access a transparent record of TechTot's development and activities, fostering trust in the system.

- **Security:** Blockchain technology enhances data security, protecting sensitive information and user interactions from unauthorized access.

## Key Data Interactions

The TechTot ecosystem involves several parallel collections, including:

1. **Info Collection:** Contains information about the blockchain state, mined blocks, and transaction details.

2. **Keys Collection:** Stores essential cryptographic keys and user identification information.

3. **Movement Collection:** Records TechTot's movements, including position and velocity.

4. **Network Collection:** Captures transaction data within the blockchain network, including sender, receiver, and block information.

These collections collectively enable secure and transparent data management, enhancing the TechTot experience.

## Setup

Follow these steps to set up the NeuroNugget stack for TechTots TechTots project:

1. Clone the NeuroNugget repository:
