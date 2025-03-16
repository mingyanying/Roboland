using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class RobotController : MonoBehaviour
{
    public GameObject root;
    public GameObject head;
    public GameObject torsoUpper;
    public GameObject torsoLower;
    public GameObject armLeftUpper;
    public GameObject armLeftLower;
    public GameObject armRightUpper;
    public GameObject armRightLower;
    public GameObject legLeftUpper;
    public GameObject legLeftLower;
    public GameObject legRightUpper;
    public GameObject legRightLower;
    public GameObject footLeft;
    public GameObject footRight;
    public GameObject handLeft;
    public GameObject handRight;

    private ConfigurableJoint headJoint;
    private ConfigurableJoint torsoUpperJoint;
    private ConfigurableJoint armLeftUpperJoint;
    private ConfigurableJoint armRightUpperJoint;
    private ConfigurableJoint legLeftUpperJoint;
    private ConfigurableJoint legRightUpperJoint;
    private HingeJoint armLeftLowerJoint;
    private HingeJoint armRightLowerJoint;
    private HingeJoint legLeftLowerJoint;
    private HingeJoint legRightLowerJoint;
    private HingeJoint footLeftJoint;
    private HingeJoint footRightJoint;

    private Rigidbody headRb;
    public Rigidbody torsoUpperRb;
    private Rigidbody torsoLowerRb;
    private Rigidbody armLeftUpperRb;
    private Rigidbody armRightUpperRb;
    private Rigidbody legLeftUpperRb;
    private Rigidbody legRightUpperRb;
    private Rigidbody armLeftLowerRb;
    private Rigidbody armRightLowerRb;
    private Rigidbody legLeftLowerRb;
    private Rigidbody legRightLowerRb;
    private Rigidbody footLeftRb;
    private Rigidbody footRightRb;
    private Rigidbody handLeftRb;
    private Rigidbody handRightRb;

    public float rotationForce = 120f;
    private Motor[] motors;
    private float[] currentFactors;

    private List<Transform> partsTransforms;
    private List<Rigidbody> partsRigidbodies;
    private Vector3[] initialPositions;
    private Quaternion[] initialRotations;

    // New method to set color of a specific part or all parts
    public void SetPartColor(GameObject part, Color color)
    {
        if (part != null)
        {
            Renderer renderer = part.GetComponent<Renderer>();
            if (renderer != null && renderer.material != null)
            {
                renderer.material.color = color; // Changes the material instance
                //Debug.Log($"Set color of {part.name} to {color}");
            }
            else
            {
                Debug.LogWarning($"No Renderer or material found on {part.name}");
            }
        }
    }

    private class Motor
    {
        public Rigidbody rb;
        public Component joint;
        public AxisType axisType;

        public void ApplyTorque(float factor, float baseForce)
        {
            if (rb == null || joint == null) return;

            Vector3 rotationAxis;
            float currentAngle = 0f;
            bool atLimit = false;

            if (joint is ConfigurableJoint configJoint)
            {
                switch (axisType)
                {
                    case AxisType.X:
                        rotationAxis = configJoint.transform.right;
                        currentAngle = GetConfigurableJointAngle(configJoint, JointDriveMode.X);
                        atLimit = IsAtAngularLimit(configJoint, JointDriveMode.X, currentAngle, factor);
                        break;
                    case AxisType.Y:
                        rotationAxis = configJoint.transform.up;
                        currentAngle = GetConfigurableJointAngle(configJoint, JointDriveMode.Y);
                        atLimit = IsAtAngularLimit(configJoint, JointDriveMode.Y, currentAngle, factor);
                        break;
                    case AxisType.Z:
                        rotationAxis = configJoint.transform.forward;
                        currentAngle = GetConfigurableJointAngle(configJoint, JointDriveMode.Z);
                        atLimit = IsAtAngularLimit(configJoint, JointDriveMode.Z, currentAngle, factor);
                        break;
                    default:
                        return;
                }
            }
            else if (joint is HingeJoint hingeJoint)
            {
                rotationAxis = hingeJoint.transform.TransformDirection(hingeJoint.axis);
                currentAngle = hingeJoint.angle;
                JointLimits limits = hingeJoint.limits;
                float tolerance = 1f;
                // Check if at limit and trying to move further in that direction
                atLimit = (currentAngle >= limits.max - tolerance && factor > 0) || (currentAngle <= limits.min + tolerance && factor < 0);
            }
            else
            {
                return;
            }

            // Only apply torque if not at limit or if torque direction would move away from limit
            if (!atLimit)
            {
                float torqueMagnitude = baseForce * factor;
                Vector3 torque = rotationAxis * torqueMagnitude;
                rb.AddTorque(torque, ForceMode.Force);
            }
            else
            {
                //Debug.LogWarning($"Joint at limit: {joint.gameObject.name}, Angle: {currentAngle}, Factor: {factor}");
            }
        }

        // Helper method to get angle from ConfigurableJoint
        private float GetConfigurableJointAngle(ConfigurableJoint joint, JointDriveMode mode)
        {
            // Get the joint's current local rotation relative to its parent
            Quaternion currentRotation = joint.transform.localRotation;

            // Calculate the rotation difference from the target rotation
            Quaternion relativeRotation = Quaternion.Inverse(joint.targetRotation) * currentRotation;
            Vector3 eulerAngles = relativeRotation.eulerAngles;

            // Normalize angles to [-180, 180]
            float angle = 0f;
            switch (mode)
            {
                case JointDriveMode.X:
                    angle = eulerAngles.x;
                    break;
                case JointDriveMode.Y:
                    angle = eulerAngles.y;
                    break;
                case JointDriveMode.Z:
                    angle = eulerAngles.z;
                    break;
            }
            if (angle > 180) angle -= 360;
            return angle;
        }

        // Helper method to check if ConfigurableJoint is at its angular limit
        private bool IsAtAngularLimit(ConfigurableJoint joint, JointDriveMode mode, float currentAngle, float factor)
        {
            float minLimit = 0f;
            float maxLimit = 0f;
            float tolerance = 1f; // Small tolerance to avoid jitter near limits

            if (mode == JointDriveMode.X)
            {
                if (joint.angularXMotion == ConfigurableJointMotion.Limited)
                {
                    minLimit = joint.lowAngularXLimit.limit;
                    maxLimit = joint.highAngularXLimit.limit;
                }
                else
                {
                    return false; // No limit if motion is free
                }
            }
            else if (mode == JointDriveMode.Y)
            {
                if (joint.angularYMotion == ConfigurableJointMotion.Limited)
                {
                    SoftJointLimit yLimit = joint.angularYLimit;
                    minLimit = -yLimit.limit; // Negative limit for Y
                    maxLimit = yLimit.limit;  // Positive limit for Y
                }
                else
                {
                    return false; // No limit if motion is free
                }
            }
            else if (mode == JointDriveMode.Z)
            {
                if (joint.angularZMotion == ConfigurableJointMotion.Limited)
                {
                    SoftJointLimit zLimit = joint.angularZLimit;
                    minLimit = -zLimit.limit; // Negative limit for Z
                    maxLimit = zLimit.limit;  // Positive limit for Z
                }
                else
                {
                    return false; // No limit if motion is free
                }
            }
            else
            {
                return false; // Invalid mode
            }

            // Check if at limit and if the torque would push it further beyond the limit
            bool atMaxLimit = (currentAngle >= maxLimit - tolerance && factor > 0);
            bool atMinLimit = (currentAngle <= minLimit + tolerance && factor < 0);
            return atMaxLimit || atMinLimit;
        }
    }

    private enum AxisType
    {
        X,
        Y,
        Z,
        Hinge
    }

    private enum JointDriveMode
    {
        X,
        Y,
        Z
    }

    void Awake()
    {
        if (root == null)
            root = gameObject;

        head = root.transform.Find("head").gameObject;
        torsoUpper = root.transform.Find("torso_upper").gameObject;
        torsoLower = root.transform.Find("torso_lower").gameObject;
        armLeftUpper = root.transform.Find("arm_left_upper").gameObject;
        armLeftLower = root.transform.Find("arm_left_lower").gameObject;
        armRightUpper = root.transform.Find("arm_right_upper").gameObject;
        armRightLower = root.transform.Find("arm_right_lower").gameObject;
        legLeftUpper = root.transform.Find("leg_left_upper").gameObject;
        legLeftLower = root.transform.Find("leg_left_lower").gameObject;
        legRightUpper = root.transform.Find("leg_right_upper").gameObject;
        legRightLower = root.transform.Find("leg_right_lower").gameObject;
        footLeft = root.transform.Find("foot_left").gameObject;
        footRight = root.transform.Find("foot_right").gameObject;
        handLeft = root.transform.Find("hand_left").gameObject;
        handRight = root.transform.Find("hand_right").gameObject;

        headJoint = head.GetComponent<ConfigurableJoint>();
        torsoUpperJoint = torsoUpper.GetComponent<ConfigurableJoint>();
        armLeftUpperJoint = armLeftUpper.GetComponent<ConfigurableJoint>();
        armRightUpperJoint = armRightUpper.GetComponent<ConfigurableJoint>();
        legLeftUpperJoint = legLeftUpper.GetComponent<ConfigurableJoint>();
        legRightUpperJoint = legRightUpper.GetComponent<ConfigurableJoint>();
        armLeftLowerJoint = armLeftLower.GetComponent<HingeJoint>();
        armRightLowerJoint = armRightLower.GetComponent<HingeJoint>();
        legLeftLowerJoint = legLeftLower.GetComponent<HingeJoint>();
        legRightLowerJoint = legRightLower.GetComponent<HingeJoint>();
        footLeftJoint = footLeft.GetComponent<HingeJoint>();
        footRightJoint = footRight.GetComponent<HingeJoint>();

        headRb = head.GetComponent<Rigidbody>();
        torsoUpperRb = torsoUpper.GetComponent<Rigidbody>();
        torsoLowerRb = torsoLower.GetComponent<Rigidbody>();
        armLeftUpperRb = armLeftUpper.GetComponent<Rigidbody>();
        armRightUpperRb = armRightUpper.GetComponent<Rigidbody>();
        legLeftUpperRb = legLeftUpper.GetComponent<Rigidbody>();
        legRightUpperRb = legRightUpper.GetComponent<Rigidbody>();
        armLeftLowerRb = armLeftLower.GetComponent<Rigidbody>();
        armRightLowerRb = armRightLower.GetComponent<Rigidbody>();
        legLeftLowerRb = legLeftLower.GetComponent<Rigidbody>();
        legRightLowerRb = legRightLower.GetComponent<Rigidbody>();
        footLeftRb = footLeft.GetComponent<Rigidbody>();
        footRightRb = footRight.GetComponent<Rigidbody>();
        handLeftRb = handLeft.GetComponent<Rigidbody>();
        handRightRb = handRight.GetComponent<Rigidbody>();

        motors = new Motor[]
        {
            new Motor { rb = headRb, joint = headJoint, axisType = AxisType.X },
            new Motor { rb = headRb, joint = headJoint, axisType = AxisType.Y },
            new Motor { rb = headRb, joint = headJoint, axisType = AxisType.Z },
            new Motor { rb = torsoUpperRb, joint = torsoUpperJoint, axisType = AxisType.X },
            new Motor { rb = torsoUpperRb, joint = torsoUpperJoint, axisType = AxisType.Y },
            new Motor { rb = torsoUpperRb, joint = torsoUpperJoint, axisType = AxisType.Z },
            new Motor { rb = armLeftUpperRb, joint = armLeftUpperJoint, axisType = AxisType.X },
            new Motor { rb = armLeftUpperRb, joint = armLeftUpperJoint, axisType = AxisType.Y },
            new Motor { rb = armLeftUpperRb, joint = armLeftUpperJoint, axisType = AxisType.Z },
            new Motor { rb = armRightUpperRb, joint = armRightUpperJoint, axisType = AxisType.X },
            new Motor { rb = armRightUpperRb, joint = armRightUpperJoint, axisType = AxisType.Y },
            new Motor { rb = armRightUpperRb, joint = armRightUpperJoint, axisType = AxisType.Z },
            new Motor { rb = legLeftUpperRb, joint = legLeftUpperJoint, axisType = AxisType.X },
            new Motor { rb = legLeftUpperRb, joint = legLeftUpperJoint, axisType = AxisType.Y },
            new Motor { rb = legLeftUpperRb, joint = legLeftUpperJoint, axisType = AxisType.Z },
            new Motor { rb = legRightUpperRb, joint = legRightUpperJoint, axisType = AxisType.X },
            new Motor { rb = legRightUpperRb, joint = legRightUpperJoint, axisType = AxisType.Y },
            new Motor { rb = legRightUpperRb, joint = legRightUpperJoint, axisType = AxisType.Z },
            new Motor { rb = armLeftLowerRb, joint = armLeftLowerJoint, axisType = AxisType.Hinge },
            new Motor { rb = armRightLowerRb, joint = armRightLowerJoint, axisType = AxisType.Hinge },
            new Motor { rb = legLeftLowerRb, joint = legLeftLowerJoint, axisType = AxisType.Hinge },
            new Motor { rb = legRightLowerRb, joint = legRightLowerJoint, axisType = AxisType.Hinge },
            new Motor { rb = footLeftRb, joint = footLeftJoint, axisType = AxisType.Hinge },
            new Motor { rb = footRightRb, joint = footRightJoint, axisType = AxisType.Hinge }
        };

        partsTransforms = new List<Transform>
        {
            head.transform, torsoUpper.transform, torsoLower.transform,
            armLeftUpper.transform, armLeftLower.transform,
            armRightUpper.transform, armRightLower.transform,
            legLeftUpper.transform, legLeftLower.transform,
            legRightUpper.transform, legRightLower.transform,
            footLeft.transform, footRight.transform
        };

        partsRigidbodies = new List<Rigidbody>
        {
            headRb, torsoUpperRb, torsoLowerRb,
            armLeftUpperRb, armLeftLowerRb,
            armRightUpperRb, armRightLowerRb,
            legLeftUpperRb, legLeftLowerRb,
            legRightUpperRb, legRightLowerRb,
            footLeftRb, footRightRb, handLeftRb, handRightRb
        };

        initialPositions = partsTransforms.Select(t => t.position).ToArray();
        initialRotations = partsTransforms.Select(t => t.rotation).ToArray();

        SetPartColor(handLeft, Color.black);
        SetPartColor(handRight, Color.black);
        SetPartColor(footLeft, Color.black);
        SetPartColor(footRight, Color.black);
    }

    void Start()
    {
        currentFactors = new float[motors.Length];
    }

    void Update()
    {
        if (Input.GetKey(KeyCode.A))
        {
            for (int i = 0; i < motors.Length; i++)
            {
                if (i % 3 == 0)
                {
                    motors[i].ApplyTorque(1, rotationForce);
                }
                else
                {
                    motors[i].ApplyTorque(-1, rotationForce);
                }
            }
        }
        
    }

    void FixedUpdate()
    {
        if (currentFactors == null) return;

        for (int i = 0; i < motors.Length; i++)
        {
            motors[i].ApplyTorque(currentFactors[i], rotationForce);
            //Debug.Log($"Applying torque to motor {i}: factor = {currentFactors[i]}");
        }
    }

    public void SetMotorFactors(float[] factors)
    {
        if (factors.Length != motors.Length)
        {
            Debug.LogError($"Factor array length ({factors.Length}) does not match the number of motors ({motors.Length}).");
            return;
        }
        currentFactors = factors;
    }

    public float[] GetJointAngles()
    {
        float[] angles = new float[24];
        angles[0] = head.transform.localEulerAngles.x;
        angles[1] = head.transform.localEulerAngles.y;
        angles[2] = head.transform.localEulerAngles.z;
        angles[3] = torsoUpper.transform.localEulerAngles.x;
        angles[4] = torsoUpper.transform.localEulerAngles.y;
        angles[5] = torsoUpper.transform.localEulerAngles.z;
        angles[6] = armLeftUpper.transform.localEulerAngles.x;
        angles[7] = armLeftUpper.transform.localEulerAngles.y;
        angles[8] = armLeftUpper.transform.localEulerAngles.z;
        angles[9] = armRightUpper.transform.localEulerAngles.x;
        angles[10] = armRightUpper.transform.localEulerAngles.y;
        angles[11] = armRightUpper.transform.localEulerAngles.z;
        angles[12] = legLeftUpper.transform.localEulerAngles.x;
        angles[13] = legLeftUpper.transform.localEulerAngles.y;
        angles[14] = legLeftUpper.transform.localEulerAngles.z;
        angles[15] = legRightUpper.transform.localEulerAngles.x;
        angles[16] = legRightUpper.transform.localEulerAngles.y;
        angles[17] = legRightUpper.transform.localEulerAngles.z;
        angles[18] = armLeftLowerJoint != null ? armLeftLowerJoint.angle : 0f;
        angles[19] = armRightLowerJoint != null ? armRightLowerJoint.angle : 0f;
        angles[20] = legLeftLowerJoint != null ? legLeftLowerJoint.angle : 0f;
        angles[21] = legRightLowerJoint != null ? legRightLowerJoint.angle : 0f;
        angles[22] = footLeftJoint != null ? footLeftJoint.angle : 0f;
        angles[23] = footRightJoint != null ? footRightJoint.angle : 0f;

        // Normalize angles and check for NaN
        for (int i = 0; i < angles.Length; i++)
        {
            if (float.IsNaN(angles[i]) || float.IsInfinity(angles[i]))
            {
                Debug.LogWarning($"NaN detected in joint angle[{i}], setting to 0.");
                angles[i] = 0f;
            }
            else if (angles[i] > 180f) angles[i] -= 360f;
            else if (angles[i] < -180f) angles[i] += 360f;
        }
        return angles;
    }

    public void ResetRobot()
    {
        //Debug.Log($"Resetting robot at time {Time.time}");
        for (int i = 0; i < partsTransforms.Count; i++)
        {
            partsTransforms[i].position = initialPositions[i];
            partsTransforms[i].rotation = initialRotations[i];
            partsRigidbodies[i].velocity = Vector3.zero;
            partsRigidbodies[i].angularVelocity = Vector3.zero;
        }
    }
}