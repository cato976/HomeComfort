using Microsoft.ML.Data;

namespace HomeComfort.ML
{
    public class FeedbackPrediction
    {
        [ColumnName(name: "PredictedLabel")]
        public bool TurnOnAC { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }

    }
}
