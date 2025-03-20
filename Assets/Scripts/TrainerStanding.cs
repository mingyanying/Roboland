using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using System;

public class TrainerStanding : MonoBehaviour
{
    public NeuralNetwork neuralNetwork;
    public RobotController[] robots; // Array to hold the two robots, assign in Inspector (set size to 2)
    public float TrainingLoopDuration = 10.0f;
    public int MaxLoops = 10000;
    public float timeScale = 2f; // Speedup factor during training, Unity seting priorty
    private float previousDistanceToTarget = float.MaxValue; // reward for move toward the target

    private List<List<float[]>> allStates = new List<List<float[]>>(); // States for each robot
    private List<List<float[]>> allActions = new List<List<float[]>>(); // Actions for each robot
    private List<List<float>> allRewards = new List<List<float>>();     // Rewards for each robot

    private float loopTimer = 0f;
    private int currentLoopCount = 0;
    private bool isTraining = false;

    void Start()
    {
        // Ensure normal speed at the start
        Time.timeScale = 1.0f;
        Time.fixedDeltaTime = 0.02f; // Default fixedDeltaTime
        Debug.Log("Simulation started at normal speed (Time.timeScale = 1.0)");

        // Initialize lists for each robot
        for (int i = 0; i < robots.Length; i++)
        {
            allStates.Add(new List<float[]>());
            allActions.Add(new List<float[]>());
            allRewards.Add(new List<float>());
        }
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.O))
        {
            foreach (var robot in robots)
            {
                if (robot != null) robot.ResetRobot();
            }
            isTraining = !isTraining; // Toggle training mode
            if (isTraining)
            {
                Time.timeScale = timeScale;
                Time.fixedDeltaTime = 0.02f * (1f / Time.timeScale); // Adjust for physics stability
                
                loopTimer = 0f;
                ClearAllData();
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
            foreach (var robot in robots)
            {
                if (robot != null) robot.ResetRobot();
            }
            loopTimer = 0f;
            ClearAllData();
            Time.timeScale = 1.0f;
            Time.fixedDeltaTime = 0.02f; // Reset to default
            Debug.Log("Training stopped! Time scale reset to 1.0");
        }
    }

    void FixedUpdate()
    {
        if (!isTraining) return;

        loopTimer += Time.fixedDeltaTime;

        // Collect data from all robots
        for (int i = 0; i < robots.Length; i++)
        {
            if (robots[i] == null)
            {
                Debug.LogWarning($"Robot at index {i} is null, skipping data collection.");
                continue;
            }

            float[] state = robots[i].GetCurrentState(1f); // Using skillNumber = 1f as placeholder
            float[] action = neuralNetwork.ForwardPass(state); // Get action based on current state
            float reward = CalculateRewardStep(robots[i]);

            // Debug state size
            if (state.Length != 180)
                Debug.LogError($"Collected state has incorrect size ({state.Length}), expected 180!");

            allStates[i].Add(state);
            allActions[i].Add(action);
            allRewards[i].Add(reward);

            // Apply action to robot
            robots[i].SetMotorFactors(action);
        }

        if (loopTimer >= TrainingLoopDuration)
        {
            TrainModel();
            
            ResetTraining();
        }
    }

    private void TrainModel()
    {
        if (neuralNetwork == null)
        {
            Debug.LogError("NeuralNetwork is null, cannot train!");
            return;
        }

        // Combine data from all robots
        float[][] combinedStates = allStates.SelectMany(list => list).ToArray();
        float[][] combinedActions = allActions.SelectMany(list => list).ToArray();
        float[] combinedRewards = allRewards.SelectMany(list => list).ToArray();

        // Check if we have any data to train on
        if (combinedStates.Length == 0 || combinedActions.Length == 0 || combinedRewards.Length == 0)
        {
            Debug.LogWarning("No data collected for training! Skipping this loop.");
            return;
        }

        // Train the neural network with combined data
        neuralNetwork.Train(combinedStates, combinedActions, combinedRewards);

        // Log training statistics
        Debug.Log($"Training loop {currentLoopCount + 1}/{MaxLoops} completed. Total samples: {combinedStates.Length}");
        float totalReward = combinedRewards.Sum();
        Debug.Log($"Total Reward across robots: {totalReward} (Sum of all rewards)");

        // Compute action diversity safely
        var allActionValues = combinedActions.SelectMany(a => a);
        if (allActionValues.Any())
        {
            float actionMean = allActionValues.Average();
            float actionStdDev = Mathf.Sqrt(allActionValues.Select(a => (a - actionMean) * (a - actionMean)).Average());
            Debug.Log($"Action Diversity (StdDev): {actionStdDev}");
        }
        else
        {
            Debug.LogWarning("No actions available to compute diversity.");
        }
    }
    

    private float CalculateRewardStep(RobotController robotController)
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
            reward += Mathf.Clamp((headHeight - 2.5f) * 100f, -100f, 2000f) * Time.fixedDeltaTime;
            
        }

        if (robotController.legLeftUpper != null)
        {
            float upperLegsHeight = robotController.legLeftUpper.transform.position.y + robotController.legRightUpper.transform.position.y;
            if (float.IsNaN(upperLegsHeight))
            {
                Debug.LogWarning("Head height is NaN, setting to 0.");
                upperLegsHeight = 0f;
            }
            reward += (upperLegsHeight - 0.8f) * 100f * Time.fixedDeltaTime;
            
        }

        if (robotController.torsoUpper != null && robotController.torsoLower != null)
        {
            float torsoUpperHeight = robotController.torsoUpper.transform.position.y;
            float torsoLowerHeight = robotController.torsoLower.transform.position.y;
            if (float.IsNaN(torsoUpperHeight)) torsoUpperHeight = 0f;
            if (float.IsNaN(torsoLowerHeight)) torsoLowerHeight = 0f;
            float avgTorsoHeight = (torsoUpperHeight + torsoLowerHeight) / 2f;
            if (avgTorsoHeight >= 1.5f)
                reward += Mathf.Clamp(avgTorsoHeight * 25f, 0f, 50f) * Time.fixedDeltaTime;
            else if (avgTorsoHeight < 1.5)
                reward -= avgTorsoHeight * 20f * Time.fixedDeltaTime;
        }

        if (robotController.footLeft != null && robotController.footRight != null)
        {
            float leftFootHeight = robotController.footLeft.transform.position.y;
            float rightFootHeight = robotController.footRight.transform.position.y;
            if (float.IsNaN(leftFootHeight)) leftFootHeight = 0f;
            if (float.IsNaN(rightFootHeight)) rightFootHeight = 0f;

            float avgHeight = (leftFootHeight + rightFootHeight)/2;

            reward += (-avgHeight + 0.3f) * 50 * Time.fixedDeltaTime; 

            
        }

        if (robotController.torsoUpper != null && robotController.footLeft != null)
        {
            float feetCenterx = Math.Abs(robotController.footLeft.transform.position.x - robotController.footRight.transform.position.x);
            float feetCenterz = Math.Abs(robotController.footLeft.transform.position.z - robotController.footRight.transform.position.z);
            float rightnessx = feetCenterx - Math.Abs(robotController.torsoUpper.transform.position.x);
            float rightnessz = feetCenterz - Math.Abs(robotController.torsoUpper.transform.position.z);
            rightnessx = Math.Abs(rightnessx);
            rightnessz = Math.Abs(rightnessz);

            if (float.IsNaN(rightnessx)) rightnessx = 0f;
            if (float.IsNaN(rightnessz)) rightnessz = 0f;
            float AvgRightness = (rightnessx + rightnessz) / 2f;
            AvgRightness = Math.Clamp(-AvgRightness + 2.3f, 0, 2.3f);
            reward += Mathf.Clamp(AvgRightness * 0.1f, 0f, 30f) * Time.fixedDeltaTime * 0.05f;
        }

        if (robotController.head != null && robotController.torsoUpper != null)
        {
            float rightnessx = Math.Abs(robotController.torsoUpper.transform.position.x) - Math.Abs(robotController.head.transform.position.x);
            float rightnessz = Math.Abs(robotController.torsoUpper.transform.position.z) - Math.Abs(robotController.head.transform.position.z);
            rightnessx = Math.Abs(rightnessx);
            rightnessz = Math.Abs(rightnessz);

            if (float.IsNaN(rightnessx)) rightnessx = 0f;
            if (float.IsNaN(rightnessz)) rightnessz = 0f;
            float AvgRightness = (rightnessx + rightnessz) / 2f;
            AvgRightness = Math.Clamp(-AvgRightness + 2.3f, 0, 2.3f);
            reward += Mathf.Clamp(AvgRightness * 15f, 0f, 100f) * Time.fixedDeltaTime;
        }
        

        if (neuralNetwork != null)
        {
            float[] rotations = neuralNetwork.GetRotation();
            float pitchPenalty = Mathf.Abs(rotations[0]) * 20f;
            float rollPenalty = Mathf.Abs(rotations[1]) * 20f;
            reward -= (pitchPenalty + rollPenalty) * Time.fixedDeltaTime;
            reward += rotations[2] * 10f * Time.fixedDeltaTime;
        }

        if (neuralNetwork != null)
        {
            float[] rotations = neuralNetwork.GetLegLeftLowerRotation();
            float pitchPenalty = Mathf.Abs(rotations[0]) * 20f;
            float rollPenalty = Mathf.Abs(rotations[1]) * 20f;
            reward -= (pitchPenalty + rollPenalty) * Time.fixedDeltaTime;
            reward += rotations[2] * 10f * Time.fixedDeltaTime;
        }

        if (neuralNetwork != null)
        {
            float[] rotations = neuralNetwork.GetLegRightLowerRotation();
            float pitchPenalty = Mathf.Abs(rotations[0]) * 20f;
            float rollPenalty = Mathf.Abs(rotations[1]) * 20f;
            reward -= (pitchPenalty + rollPenalty) * Time.fixedDeltaTime;
            reward += rotations[2] * 10f * Time.fixedDeltaTime;
        }


        if (robotController.legLeftLower != null && robotController.legRightLower != null)
        {
            float leftLegAngle = robotController.GetJointAngles()[20];
            float rightLegAngle = robotController.GetJointAngles()[21];
            float angleDiff = Mathf.Abs(leftLegAngle - rightLegAngle);
            if (angleDiff < 30f)
                reward += (30f - angleDiff) * 1f * Time.fixedDeltaTime;

            // reward += (-leftLegAngle - rightLegAngle + 60f) * 1f * Time.fixedDeltaTime; 
        }

        if (float.IsNaN(reward) || float.IsInfinity(reward))
        {
            Debug.LogWarning("Reward is NaN or Infinity, setting to 0.");
            reward = 0f;
        }
        
        if(robotController.head != null && robotController.footLeft!= null && robotController.footRight!= null)
        {
            Vector3 headPosition = robotController.head.transform.position;
            Vector3 footPosition = robotController.footLeft.transform.position;
            Vector3 footPosition_1 = robotController.footRight.transform.position;
            float currentDis = Vector3.Distance(headPosition, footPosition) + Vector3.Distance(headPosition, footPosition_1);
            reward += (currentDis / 2 - 2.9f) * 50f * Time.fixedDeltaTime; 
        }

        Transform target = TargetManager.Instance?.GetTarget();

        if (TargetManager.Instance?.target != null)
        {

            // Calculate current distance to the target
            Vector3 robotPosition = robotController.torsoUpper.transform.position;
            Vector3 targetPosition = target.position;
            float currentDistanceToTarget = Vector3.Distance(robotPosition, targetPosition);
            float[] visionData = robotController.GetVisionData();

            // Reward for having the target in the central field of view
            
            for (int i = 64; i < 128; i++ )
            {
                if (visionData[i] == 1)                   
                    reward += 2 * Time.fixedDeltaTime;
            }
            

            if (visionData[91] == 1 ||
                visionData[92] == 1 ||
                visionData[99] == 1 ||
                visionData[100] == 1)
            {
                reward += 10f * Time.fixedDeltaTime;
            }
        /*
            // Reward for getting closer to the target
            if (previousDistanceToTarget != float.MaxValue)
            {
                float distanceDelta = previousDistanceToTarget - currentDistanceToTarget;
                if (distanceDelta > 0) // Getting closer
                {
                    float distanceReward = distanceDelta * 10f * Time.fixedDeltaTime;
                    reward += distanceReward;
                }
                else if (distanceDelta < 0) // Moving away
                {
                    float distancePenalty = distanceDelta * 10f * Time.fixedDeltaTime;
                    reward += distancePenalty;
                }
            }

            previousDistanceToTarget = currentDistanceToTarget;*/
        }



        reward = Mathf.Clamp(reward, -200f, 400f);
        return reward;
    }

    private void ResetTraining()
    {
        foreach (var robot in robots)
        {
            robot.ResetRobot();
        }
        ClearAllData();
        
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

    private void ClearAllData()
    {
        foreach (var list in allStates) list.Clear();
        foreach (var list in allActions) list.Clear();
        foreach (var list in allRewards) list.Clear();
    }
}