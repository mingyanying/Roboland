using UnityEngine;
using System.Collections.Generic;
using System.Linq;

public class TrainerStanding : MonoBehaviour
{
    public NeuralNetwork neuralNetwork;
    public RobotController robotController;
    public float TrainingLoopDuration = 10.0f;
    public int MaxLoops = 10000;
    public float timeScale = 2f; // Speedup factor during training

    private List<float[]> states = new List<float[]>();
    private List<float[]> actions = new List<float[]>();
    private List<float> rewards = new List<float>();

    private float loopTimer = 0f;
    private int currentLoopCount = 0;
    private bool isTraining = false;

    void Start()
    {
        // Ensure normal speed at the start
        Time.timeScale = 1.0f;
        Time.fixedDeltaTime = 0.02f; // Default fixedDeltaTime
        Debug.Log("Simulation started at normal speed (Time.timeScale = 1.0)");
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.O))
        {
            robotController.ResetRobot();
            isTraining = !isTraining; // Toggle training mode
            if (isTraining)
            {
                Time.timeScale = timeScale;
                Time.fixedDeltaTime = 0.02f * (1f / Time.timeScale); // Adjust for physics stability
                currentLoopCount = 0;
                loopTimer = 0f;
                states.Clear();
                actions.Clear();
                rewards.Clear();
                Debug.Log($"Training started! Time scale set to {Time.timeScale}, fixedDeltaTime set to {Time.fixedDeltaTime}");
            }
            else
            {
                Time.timeScale = 1.0f;
                Time.fixedDeltaTime = 0.02f; // Reset to default
                Debug.Log("Training stopped! Time scale reset to 1.0");
            }
        }
        if (Input.GetKey(KeyCode.P))
        {
            isTraining = false;
            robotController.ResetRobot();
            loopTimer = 0f;
            states.Clear();
            actions.Clear();
            rewards.Clear();
            Time.timeScale = 1.0f;
            Time.fixedDeltaTime = 0.02f; // Reset to default
            Debug.Log("Training stopped! Time scale reset to 1.0");
        }
    }

    void FixedUpdate()
    {
        if (!isTraining) return;

        loopTimer += Time.fixedDeltaTime;

        if (neuralNetwork != null)
        {
            float[] state = neuralNetwork.GetCurrentState();
            float[] action = neuralNetwork.GetLastOutput();
            float reward = CalculateRewardStep();

            states.Add(state);
            actions.Add(action);
            rewards.Add(reward);
        }
        else
        {
            Debug.LogError("NeuralNetwork is null, cannot collect data!");
        }

        if (loopTimer >= TrainingLoopDuration)
        {
            neuralNetwork.Train(states.ToArray(), actions.ToArray(), rewards.ToArray());
            Debug.Log($"Training loop {currentLoopCount + 1}/{MaxLoops} completed. States: {states.Count}, Rewards: {rewards.Count}");

            float[] lastActions = neuralNetwork.GetLastOutput();
            float actionMean = lastActions.Average();
            float actionStdDev = Mathf.Sqrt(lastActions.Select(a => (a - actionMean) * (a - actionMean)).Average());
            Debug.Log($"Action Diversity (StdDev): {actionStdDev}");

            float totalReward = 0f;
            foreach (float r in rewards)
            {
                totalReward += r;
            }
            Debug.Log($"Training Loop {currentLoopCount + 1} Total Reward: {totalReward}");

            if (robotController != null)
            {
                robotController.ResetRobot();
            }
            else
            {
                Debug.LogError("RobotController is null, reset failed!");
            }

            states.Clear();
            actions.Clear();
            rewards.Clear();
            loopTimer = 0f;

            currentLoopCount++;
            if (currentLoopCount >= MaxLoops)
            {
                Debug.Log($"Completed {MaxLoops} training loops. Stopping training.");
                isTraining = false;
                Time.timeScale = 1.0f;
                Time.fixedDeltaTime = 0.02f; // Reset to default
                Debug.Log("Time scale reset to 1.0 after training completion");
            }
        }
    }

    private float CalculateRewardStep()
    {
        float reward = 0f;

        if (robotController.head != null)
        {
            float headHeight = robotController.head.transform.position.y;
            if (float.IsNaN(headHeight))
            {
                Debug.LogWarning("Head height is NaN, setting to 0.");
                headHeight = 0f;
            }
            reward += Mathf.Clamp(headHeight * 20f, 0f, 50f) * Time.fixedDeltaTime;
            if (headHeight < 0.5f)
            {
                reward -= 10f * Time.fixedDeltaTime;
            }
        }

        if (robotController.torsoUpper != null && robotController.torsoLower != null)
        {
            float torsoUpperHeight = robotController.torsoUpper.transform.position.y;
            float torsoLowerHeight = robotController.torsoLower.transform.position.y;
            if (float.IsNaN(torsoUpperHeight)) torsoUpperHeight = 0f;
            if (float.IsNaN(torsoLowerHeight)) torsoLowerHeight = 0f;
            float avgTorsoHeight = (torsoUpperHeight + torsoLowerHeight) / 2f;
            reward += Mathf.Clamp(avgTorsoHeight * 15f, 0f, 30f) * Time.fixedDeltaTime;
        }

        if (robotController.footLeft != null && robotController.footRight != null)
        {
            float leftFootHeight = robotController.footLeft.transform.position.y;
            float rightFootHeight = robotController.footRight.transform.position.y;
            if (float.IsNaN(leftFootHeight)) leftFootHeight = 0f;
            if (float.IsNaN(rightFootHeight)) rightFootHeight = 0f;

            if (leftFootHeight <= 0.2f)
                reward += 15f * Time.fixedDeltaTime;
            else
                reward -= Mathf.Clamp((leftFootHeight - 0.2f) * 5f, 0f, 20f) * Time.fixedDeltaTime;

            if (rightFootHeight <= 0.2f)
                reward += 15f * Time.fixedDeltaTime;
            else
                reward -= Mathf.Clamp((rightFootHeight - 0.2f) * 5f, 0f, 20f) * Time.fixedDeltaTime;
        }

        if (neuralNetwork != null)
        {
            float[] rotations = neuralNetwork.GetRotation();
            float pitchPenalty = Mathf.Abs(rotations[0]) * 20f;
            float rollPenalty = Mathf.Abs(rotations[1]) * 20f;
            reward -= (pitchPenalty + rollPenalty) * Time.fixedDeltaTime;
            reward += rotations[2] * 30f * Time.fixedDeltaTime;
        }

        if (robotController.legLeftLower != null && robotController.legRightLower != null)
        {
            float leftLegAngle = robotController.GetJointAngles()[20];
            float rightLegAngle = robotController.GetJointAngles()[21];
            float angleDiff = Mathf.Abs(leftLegAngle - rightLegAngle);
            if (angleDiff < 30f)
                reward += (30f - angleDiff) * 0.5f * Time.fixedDeltaTime;
        }

        if (float.IsNaN(reward) || float.IsInfinity(reward))
        {
            Debug.LogWarning("Reward is NaN or Infinity, setting to 0.");
            reward = 0f;
        }
        reward = Mathf.Clamp(reward, -50f, 100f);
        return reward;
    }
}