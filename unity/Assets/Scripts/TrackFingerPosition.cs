using System.Collections;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.Windows.Speech;

public class TrackFingerPosition : MonoBehaviour
{
    private KeywordRecognizer keywordRecognizer;
    private Dictionary<string, System.Action> actions = new Dictionary<string, System.Action>();

    public GameObject Finger_Mark_Prefab;
    public Material Finger_Mark_Material;

    void Start()
    {
        actions.Add("Set", SetPosition);
        keywordRecognizer = new KeywordRecognizer(actions.Keys.ToArray());
        keywordRecognizer.OnPhraseRecognized += RecognizedSpeech;
        keywordRecognizer.Start();
    }

    private void RecognizedSpeech(PhraseRecognizedEventArgs speech)
    {
        System.Action keywordAction;
        if (actions.TryGetValue(speech.text, out keywordAction))
        {
            keywordAction.Invoke();
        }
    }

    private void SetPosition()
    {
        this.transform.parent = null;
        Transform CursorTransform = transform;
        Vector3 position = CursorTransform.position;
        Debug.Log("Set finger position: " + position.ToString("F6"));
        GameObject fingerMark = Instantiate(Finger_Mark_Prefab, position, Quaternion.identity);
        Renderer renderer = fingerMark.GetComponent<Renderer>();
        if (renderer != null && Finger_Mark_Material != null)
        {
            renderer.material = Finger_Mark_Material;
        }
        GlobalVariables.mark_cnt++;
        fingerMark.name = "fingerSphere" + GlobalVariables.mark_cnt.ToString();
    }
}