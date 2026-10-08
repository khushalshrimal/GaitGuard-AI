# GaitGuard AI — Phase 15 Blinded Double-Annotation & Adjudication Workflow

## 1. Workflow Architecture

```mermaid
graph TD
    A[Cattle Walkway Video Input] --> B[Blinded Annotator 1: Sprecher Score]
    A --> C[Blinded Annotator 2: Sprecher Score]
    B --> D{Compare Binary Reference Labels}
    C --> D
    D -->|Agreement: A1 == A2| E[Consensus Reference Label Recorded]
    D -->|Disagreement: A1 != A2| F[Third Reviewer Adjudication]
    F --> G[Adjudicated Reference Label Recorded]
    E --> H[External Validation Manifest]
    G --> H
```

## 2. Blinded Annotation Rules
1. **Model Output Isolation**: Annotators MUST NOT see GaitGuard predictions, probabilities, SHAP feature attributions, or Quality Gate scores.
2. **Peer Isolation**: Annotator 1 and Annotator 2 MUST NOT see each other's score entries prior to consensus submission.
3. **Adjudication Rule**: In cases of score disagreement between Annotator 1 and Annotator 2, a third senior veterinarian adjudicates the final reference score without overwriting original raw evaluator entries.
