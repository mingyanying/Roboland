using System.Collections.Generic;
using UnityEngine;

public class PlaneManager : MonoBehaviour
{
    public static PlaneManager Instance { get; private set; }

    [SerializeField] private GameObject[] planes; // Assign planes in the Inspector
    [SerializeField] private float planeDynamicFriction = 0f; // Set to 0 to ensure robot's friction dominates
    [SerializeField] private float planeStaticFriction = 0f;
    [SerializeField] private float planeBounciness = 0f;

    private void Awake()
    {
        // Singleton pattern to ensure only one PlaneManager exists
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
            return;
        }

        // Find planes if not assigned
        if (planes == null || planes.Length == 0)
        {
            planes = GameObject.FindGameObjectsWithTag("Ground");
            if (planes.Length == 0)
            {
                Debug.LogWarning("PlaneManager: No planes found with tag 'Ground'. Please assign planes in the Inspector or tag them as 'Ground'.");
            }
        }

        // Set up physics materials for all planes
        SetupPlaneMaterials();
    }

    private void SetupPlaneMaterials()
    {
        // Create a shared physics material for all planes
        PhysicMaterial planeMaterial = new PhysicMaterial("PlaneMaterial")
        {
            dynamicFriction = planeDynamicFriction,
            staticFriction = planeStaticFriction,
            bounciness = planeBounciness,
            frictionCombine = PhysicMaterialCombine.Maximum, // Ensure robot's friction dominates
            bounceCombine = PhysicMaterialCombine.Average
        };

        // Assign the material to each plane's collider
        foreach (GameObject plane in planes)
        {
            if (plane == null)
            {
                Debug.LogWarning("PlaneManager: One of the assigned planes is null. Check the Inspector.");
                continue;
            }

            Collider planeCollider = plane.GetComponent<Collider>();
            if (planeCollider == null)
            {
                Debug.LogWarning($"PlaneManager: Plane '{plane.name}' has no Collider component. Adding a MeshCollider.");
                planeCollider = plane.AddComponent<MeshCollider>();
            }

            planeCollider.material = planeMaterial;
            Debug.Log($"PlaneManager: Assigned physics material to plane '{plane.name}' with Dynamic Friction: {planeDynamicFriction}, Static Friction: {planeStaticFriction}");
        }
    }

    // Optional: Method to update friction values at runtime
    public void UpdatePlaneFriction(float dynamicFriction, float staticFriction)
    {
        planeDynamicFriction = dynamicFriction;
        planeStaticFriction = staticFriction;
        SetupPlaneMaterials(); // Reapply the materials with updated values
    }

    // Optional: Get the physics material for external use
    public PhysicMaterial GetPlaneMaterial()
    {
        return new PhysicMaterial("PlaneMaterial")
        {
            dynamicFriction = planeDynamicFriction,
            staticFriction = planeStaticFriction,
            bounciness = planeBounciness,
            frictionCombine = PhysicMaterialCombine.Maximum,
            bounceCombine = PhysicMaterialCombine.Average
        };
    }
}