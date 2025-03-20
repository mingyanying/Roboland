using System.Collections.Generic;
using UnityEngine;

public class TargetManager : MonoBehaviour
{
    public GameObject target; // Assign in the Inspector
    private static TargetManager instance;

    void Awake()
    {
        // Singleton pattern to ensure only one TargetManager exists
        if (instance == null)
        {
            instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }

        SetPartColor(target, Color.red);

        // Validate target assignment
        if (target == null)
        {
            Debug.LogError("TargetManager: Target is not assigned! Please assign a target in the Inspector.");
        }
    }

    private void SetPartColor(GameObject part, Color color)
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

    public static TargetManager Instance
    {
        get { return instance; }
    }

    public Transform GetTarget()
    {
        if (target == null)
        {
            Debug.LogWarning("TargetManager: Target is null! Returning null.");
        }
        return target.transform;
    }

    public Vector3 GetTargetPosition()
    {
        if (target == null)
        {
            Debug.LogWarning("TargetManager: Target is null! Returning Vector3.zero.");
            return Vector3.zero;
        }
        return target.transform.position;
    }

    // Optional: Add logic to move or spawn the target
    public void MoveTargetTo(Vector3 newPosition)
    {
        if (target != null)
        {
            target.transform.position = newPosition;
            Debug.Log($"TargetManager: Moved target to {newPosition}");
        }
        else
        {
            Debug.LogWarning("TargetManager: Cannot move target because it is null!");
        }
    }
}