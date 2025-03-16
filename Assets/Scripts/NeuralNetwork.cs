using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;
using System.Linq;

public class NeuralNetwork : MonoBehaviour
{
    private const int InputSize = 52;
    private const int HiddenSize1 = 128;
    private const int HiddenSize2 = 64;
    private const int OutputSize = 24;

    private float[,] weightsInputHidden1;
    private float[] biasesHidden1;
    private float[,] weightsHidden1Hidden2;
    private float[] biasesHidden2;
    private float[,] weightsHidden2Output;
    private float[] biasesOutput;

    private float[] jointAngles = new float[24];
    private float[] jointSpeeds = new float[24];
    private float[] previousJointAngles = new float[24];
    private float skillNumber = 1f;

    public RobotController robotController;

    private float timer = 0f;
    private const float InferenceInterval = 0.02f;

    private float[] lastOutput;

    private Rigidbody torsoUpperRb;

    private int trainingLoopCounter = 0; // Track training loops for perturbation

    void Awake()
    {
        weightsInputHidden1 = new float[InputSize, HiddenSize1];
        biasesHidden1 = new float[HiddenSize1];
        weightsHidden1Hidden2 = new float[HiddenSize1, HiddenSize2];
        biasesHidden2 = new float[HiddenSize2];
        weightsHidden2Output = new float[HiddenSize2, OutputSize];
        biasesOutput = new float[OutputSize];

        if (robotController != null && robotController.torsoUpper != null)
        {
            torsoUpperRb = robotController.torsoUpper.GetComponent<Rigidbody>();
            if (torsoUpperRb == null)
            {
                Debug.LogError("Upper torso Rigidbody not found!");
            }
        }
        else
        {
            Debug.LogError("RobotController or torsoUpper reference is missing!");
        }

        LoadModel();
    }

    void Start()
    {
        if (robotController == null)
        {
            Debug.LogError("RobotController reference is missing!");
            return;
        }
        Debug.Log("RobotController reference found: " + robotController.name);
        CollectJointData();
        for (int i = 0; i < 24; i++)
        {
            previousJointAngles[i] = jointAngles[i];
        }
    }

    void FixedUpdate()
    {
        timer += Time.fixedDeltaTime;
        if (timer >= InferenceInterval)
        {
            CollectJointData();
            float[] inputs = PrepareInputs();
            lastOutput = ForwardPass(inputs);
            if (robotController != null)
            {
                robotController.SetMotorFactors(lastOutput);
            }
            else
            {
                Debug.LogError("robotController is null during inference!");
            }
            timer = 0f;
        }
    }

    private string GetModelPath()
    {
        string modelsDir = @"C:\Local Documents\Robust AI\Assets\Models";
        if (!Directory.Exists(modelsDir))
        {
            try
            {
                Directory.CreateDirectory(modelsDir);
                Debug.Log($"Created directory: {modelsDir}");
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Failed to create directory {modelsDir}. Error: {e.Message}");
                return null;
            }
        }
        return Path.Combine(modelsDir, "modelG1.json");
    }

    private void LoadModel()
    {
        string path = GetModelPath();
        if (string.IsNullOrEmpty(path))
        {
            Debug.LogError("Model path is invalid. Cannot load or save model.");
            return;
        }

        if (File.Exists(path))
        {
            try
            {
                string json = File.ReadAllText(path);
                ModelData modelData = JsonUtility.FromJson<ModelData>(json);

                weightsInputHidden1 = modelData.weightsInputHidden1.To2DArray();
                biasesHidden1 = modelData.biasesHidden1;
                weightsHidden1Hidden2 = modelData.weightsHidden1Hidden2.To2DArray();
                biasesHidden2 = modelData.biasesHidden2;
                weightsHidden2Output = modelData.weightsHidden2Output.To2DArray();
                biasesOutput = modelData.biasesOutput;
                skillNumber = modelData.skillNumber;

                Debug.Log($"Model loaded from {path}. Skill number: {skillNumber}.");
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Failed to load model from {path}. Error: {e.Message}. Initializing new model.");
                InitializeAndSaveModel();
            }
        }
        else
        {
            Debug.LogWarning($"No model found at {path}. Initializing new model.");
            InitializeAndSaveModel();
        }
    }

    private void InitializeAndSaveModel()
    {
        InitializeWeights(weightsInputHidden1);
        InitializeBiases(biasesHidden1);
        InitializeWeights(weightsHidden1Hidden2);
        InitializeBiases(biasesHidden2);
        InitializeWeights(weightsHidden2Output);
        InitializeBiases(biasesOutput);
        SaveModel();
    }

    private void SaveModel()
    {
        string path = GetModelPath();
        if (string.IsNullOrEmpty(path))
        {
            Debug.LogError("Model path is invalid. Cannot save model.");
            return;
        }

        var modelData = new ModelData
        {
            weightsInputHidden1 = new SerializableMatrix(weightsInputHidden1),
            biasesHidden1 = biasesHidden1,
            weightsHidden1Hidden2 = new SerializableMatrix(weightsHidden1Hidden2),
            biasesHidden2 = biasesHidden2,
            weightsHidden2Output = new SerializableMatrix(weightsHidden2Output),
            biasesOutput = biasesOutput,
            skillNumber = skillNumber
        };

        try
        {
            string json = JsonUtility.ToJson(modelData, true);
            File.WriteAllText(path, json);
            Debug.Log($"Model saved to {path}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to save model to {path}. Error: {e.Message}");
        }
    }

    private void CollectJointData()
    {
        float[] currentAngles = robotController.GetJointAngles();
        for (int i = 0; i < 24; i++)
        {
            jointAngles[i] = currentAngles[i];
            float deltaAngle = AngleDelta(jointAngles[i], previousJointAngles[i]);
            jointSpeeds[i] = deltaAngle / InferenceInterval;
            previousJointAngles[i] = jointAngles[i];
        }
    }

    private float AngleDelta(float current, float previous)
    {
        float delta = current - previous;
        while (delta > 180) delta -= 360;
        while (delta < -180) delta += 360;
        return delta;
    }

    private float[] PrepareInputs()
    {
        float[] inputs = new float[InputSize];
        int index = 0;

        for (int i = 0; i < 24; i++)
        {
            inputs[index++] = Mathf.Clamp(jointAngles[i] / 180f, -1f, 1f);
        }

        float maxSpeed = 360f;
        for (int i = 0; i < 24; i++)
        {
            inputs[index++] = Mathf.Clamp(jointSpeeds[i] / maxSpeed, -1f, 1f);
        }

        inputs[index++] = Mathf.Clamp(skillNumber, -1f, 1f);

        if (torsoUpperRb != null && robotController.torsoUpper != null)
        {
            float[] rotations = GetRotation();
            inputs[index++] = rotations[0];
            inputs[index++] = rotations[1];
            inputs[index++] = rotations[2];
        }
        else
        {
            inputs[index++] = 0f;
            inputs[index++] = 0f;
            inputs[index++] = 0f;
            Debug.LogWarning("Torso data unavailable, using zero inputs for pitch, roll, and directional factor.");
        }

        return inputs;
    }

    public float[] GetRotation()
    {
        float[] rotations = new float[3];
        Vector3 totalAcceleration = Physics.gravity + (torsoUpperRb.velocity - torsoUpperRb.GetPointVelocity(torsoUpperRb.position)) / Time.fixedDeltaTime;

        // Check for near-zero magnitude to prevent NaN
        if (totalAcceleration.magnitude < 1e-5f)
        {
            Debug.LogWarning("Total acceleration near zero, using default gravity direction.");
            totalAcceleration = Physics.gravity; // Fallback to gravity
        }

        Vector3 gravityDir = -totalAcceleration.normalized;
        Vector3 localUp = robotController.torsoUpper.transform.up;

        Vector3 forward = robotController.torsoUpper.transform.forward;
        Vector3 gravityOnXZ = Vector3.ProjectOnPlane(gravityDir, localUp);
        float pitch = Vector3.SignedAngle(localUp, gravityOnXZ, forward);
        pitch = Mathf.Clamp(pitch, -90f, 90f);
        rotations[0] = pitch / 90f;

        Vector3 right = robotController.torsoUpper.transform.right;
        Vector3 gravityOnYZ = Vector3.ProjectOnPlane(gravityDir, right);
        float roll = Vector3.SignedAngle(localUp, gravityOnYZ, right);
        roll = Mathf.Clamp(roll, -90f, 90f);
        rotations[1] = roll / 90f;

        float dotProduct = Vector3.Dot(gravityDir, localUp);
        rotations[2] = Mathf.Sign(dotProduct);

        // Final check for NaN in rotations
        for (int i = 0; i < 3; i++)
        {
            if (float.IsNaN(rotations[i]) || float.IsInfinity(rotations[i]))
            {
                Debug.LogWarning($"NaN detected in rotations[{i}], setting to 0.");
                rotations[i] = 0f;
            }
        }

        return rotations;
    }

    private float[] ForwardPass(float[] inputs)
    {
        float[] hidden1 = new float[HiddenSize1];
        for (int i = 0; i < HiddenSize1; i++)
        {
            hidden1[i] = biasesHidden1[i];
            for (int j = 0; j < InputSize; j++)
                hidden1[i] += inputs[j] * weightsInputHidden1[j, i];
            hidden1[i] = Mathf.Max(0, hidden1[i]);
        }

        float[] hidden2 = new float[HiddenSize2];
        for (int i = 0; i < HiddenSize2; i++)
        {
            hidden2[i] = biasesHidden2[i];
            for (int j = 0; j < HiddenSize1; j++)
                hidden2[i] += hidden1[j] * weightsHidden1Hidden2[j, i];
            hidden2[i] = Mathf.Max(0, hidden2[i]);
        }

        float[] outputs = new float[OutputSize];
        for (int i = 0; i < OutputSize; i++)
        {
            outputs[i] = biasesOutput[i];
            for (int j = 0; j < HiddenSize2; j++)
            {
                outputs[i] += hidden2[j] * weightsHidden2Output[j, i];
            }
            outputs[i] = (float)Math.Tanh(outputs[i]);
        }
        return outputs;
    }

    private void InitializeWeights(float[,] weights)
    {
        int rows = weights.GetLength(0);
        int cols = weights.GetLength(1);
        float limit = Mathf.Sqrt(6f / (rows + cols));

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                weights[i, j] = UnityEngine.Random.Range(-limit, limit);
            }
        }
    }

    private void InitializeBiases(float[] biases)
    {
        for (int i = 0; i < biases.Length; i++)
        {
            biases[i] = 0f;
        }
    }

    // **Change 6: Gaussian noise helper method**
    private float GaussianNoise(float stdDev)
    {
        float u1 = UnityEngine.Random.value;
        float u2 = UnityEngine.Random.value;
        float z = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Cos(2.0f * Mathf.PI * u2);
        return z * stdDev;
    }

    private void PerturbWeights(float perturbationScale)
    {
        
        for (int i = 0; i < InputSize; i++)
        {
            float noise = GaussianNoise(perturbationScale);
            for (int j = 0; j < HiddenSize1; j++)
            {
                weightsInputHidden1[i, j] += noise;
            }
        }

        for (int i = 0; i < HiddenSize1; i++)
        {
            float noise = GaussianNoise(perturbationScale);
            for (int j = 0; j < HiddenSize2; j++)
            {
                weightsHidden1Hidden2[i, j] += noise;
            }
        }

        for (int i = 0; i < HiddenSize2; i++)
        {
            float noise = GaussianNoise(perturbationScale);
            for (int j = 0; j < OutputSize; j++)
            {
                weightsHidden2Output[i, j] += noise;
            }
        }

        Debug.Log($"Perturbed weights with scale {perturbationScale}");
    }

    public float[] GetCurrentState()
    {
        return PrepareInputs();
    }

    public float[] GetLastOutput()
    {
        return lastOutput ?? new float[OutputSize];
    }

    public void Train(float[][] states, float[][] actions, float[] rewards)
    {
        float[] returns = ComputeReturns(rewards, gamma: 0.99f);
        float learningRate = 0.005f;
        UpdateWeights(states, actions, returns, learningRate);

        trainingLoopCounter++;
        if (trainingLoopCounter % 10 == 0)
        {
            float perturbationScale = 0.05f;
            PerturbWeights(perturbationScale);
        }

        SaveModel();
        Debug.Log($"Neural network weights updated and model saved at loop {trainingLoopCounter}");
    }

    private float[] ComputeReturns(float[] rewards, float gamma)
    {
        float[] returns = new float[rewards.Length];
        float runningReturn = 0f;

        for (int t = rewards.Length - 1; t >= 0; t--)
        {
            if (float.IsNaN(rewards[t]) || float.IsInfinity(rewards[t]))
            {
                Debug.LogWarning($"NaN in rewards[{t}], setting to 0.");
                rewards[t] = 0f;
            }
            runningReturn = rewards[t] + gamma * runningReturn;
            returns[t] = runningReturn;
        }

        float mean = returns.Average();
        if (float.IsNaN(mean))
        {
            Debug.LogWarning("Mean of returns is NaN, setting returns to zero.");
            return new float[returns.Length]; // Return zeros if mean is NaN
        }

        float std = Mathf.Sqrt(returns.Select(r => (r - mean) * (r - mean)).Average()) + 1e-5f;
        if (float.IsNaN(std) || std <= 1e-5f)
        {
            Debug.LogWarning("Std of returns is invalid, skipping normalization.");
            return returns; // Skip normalization if std is invalid
        }

        for (int t = 0; t < returns.Length; t++)
        {
            returns[t] = (returns[t] - mean) / std;
            if (float.IsNaN(returns[t]) || float.IsInfinity(returns[t]))
                returns[t] = 0f;
        }
        return returns;
    }

    private void UpdateWeights(float[][] states, float[][] actions, float[] returns, float learningRate)
    {
        for (int t = 0; t < states.Length; t++)
        {
            float[] state = states[t];
            float[] action = actions[t];
            float G = returns[t];

            if (float.IsNaN(G) || float.IsInfinity(G))
            {
                Debug.LogWarning($"Invalid return G at timestep {t}, skipping.");
                continue;
            }

            float[] hidden1 = new float[HiddenSize1];
            float[] hidden1PreActivation = new float[HiddenSize1];
            for (int i = 0; i < HiddenSize1; i++)
            {
                hidden1PreActivation[i] = biasesHidden1[i];
                for (int j = 0; j < InputSize; j++)
                    hidden1PreActivation[i] += state[j] * weightsInputHidden1[j, i];
                hidden1[i] = Mathf.Max(0, hidden1PreActivation[i]);
            }

            float[] hidden2 = new float[HiddenSize2];
            float[] hidden2PreActivation = new float[HiddenSize2];
            for (int i = 0; i < HiddenSize2; i++)
            {
                hidden2PreActivation[i] = biasesHidden2[i];
                for (int j = 0; j < HiddenSize1; j++)
                    hidden2PreActivation[i] += hidden1[j] * weightsHidden1Hidden2[j, i];
                hidden2[i] = Mathf.Max(0, hidden2PreActivation[i]);
            }

            float[] outputs = new float[OutputSize];
            for (int i = 0; i < OutputSize; i++)
            {
                outputs[i] = biasesOutput[i];
                for (int j = 0; j < HiddenSize2; j++)
                    outputs[i] += hidden2[j] * weightsHidden2Output[j, i];
                outputs[i] = (float)Math.Tanh(outputs[i]);
            }

            float[] outputGradients = new float[OutputSize];
            for (int i = 0; i < OutputSize; i++)
            {
                outputGradients[i] = (action[i] - outputs[i]) * G;
                if (float.IsNaN(outputGradients[i]) || float.IsInfinity(outputGradients[i]))
                    outputGradients[i] = 0f;
            }

            float[] hidden2Gradients = new float[HiddenSize2];
            for (int i = 0; i < HiddenSize2; i++)
            {
                hidden2Gradients[i] = 0f;
                for (int j = 0; j < OutputSize; j++)
                {
                    hidden2Gradients[i] += outputGradients[j] * weightsHidden2Output[i, j];
                    float weightUpdate = learningRate * hidden2[i] * outputGradients[j];
                    if (!float.IsNaN(weightUpdate) && !float.IsInfinity(weightUpdate))
                        weightsHidden2Output[i, j] += weightUpdate;
                }
                float biasUpdate = learningRate * hidden2Gradients[i] * (hidden2PreActivation[i] > 0 ? 1f : 0f);
                if (!float.IsNaN(biasUpdate) && !float.IsInfinity(biasUpdate))
                    biasesHidden2[i] += biasUpdate;
            }

            for (int i = 0; i < OutputSize; i++)
            {
                float biasUpdate = learningRate * outputGradients[i];
                if (!float.IsNaN(biasUpdate) && !float.IsInfinity(biasUpdate))
                    biasesOutput[i] += biasUpdate;
            }

            float[] hidden1Gradients = new float[HiddenSize1];
            for (int i = 0; i < HiddenSize1; i++)
            {
                hidden1Gradients[i] = 0f;
                for (int j = 0; j < HiddenSize2; j++)
                {
                    float grad = hidden2Gradients[j] * (hidden2PreActivation[j] > 0 ? 1f : 0f);
                    hidden1Gradients[i] += grad * weightsHidden1Hidden2[i, j];
                    float weightUpdate = learningRate * hidden1[i] * grad;
                    if (!float.IsNaN(weightUpdate) && !float.IsInfinity(weightUpdate))
                        weightsHidden1Hidden2[i, j] += weightUpdate;
                }
                float biasUpdate = learningRate * hidden1Gradients[i] * (hidden1PreActivation[i] > 0 ? 1f : 0f);
                if (!float.IsNaN(biasUpdate) && !float.IsInfinity(biasUpdate))
                    biasesHidden1[i] += biasUpdate;
            }

            for (int i = 0; i < InputSize; i++)
            {
                for (int j = 0; j < HiddenSize1; j++)
                {
                    float grad = hidden1Gradients[j] * (hidden1PreActivation[j] > 0 ? 1f : 0f);
                    float weightUpdate = learningRate * state[i] * grad;
                    if (!float.IsNaN(weightUpdate) && !float.IsInfinity(weightUpdate))
                        weightsInputHidden1[i, j] += weightUpdate;
                }
            }
        }
    }

    [System.Serializable]
    private class SerializableMatrix
    {
        public float[] data;
        public int rows;
        public int cols;

        public SerializableMatrix(float[,] array)
        {
            rows = array.GetLength(0);
            cols = array.GetLength(1);
            data = new float[rows * cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    data[i * cols + j] = array[i, j];
        }

        public float[,] To2DArray()
        {
            float[,] array = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    array[i, j] = data[i * cols + j];
            return array;
        }
    }

    [System.Serializable]
    private class ModelData
    {
        public SerializableMatrix weightsInputHidden1;
        public float[] biasesHidden1;
        public SerializableMatrix weightsHidden1Hidden2;
        public float[] biasesHidden2;
        public SerializableMatrix weightsHidden2Output;
        public float[] biasesOutput;
        public float skillNumber;
    }
}