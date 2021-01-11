using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HomeComfort.ML
{
    public class FeedbackHeatPrediction
    {
        [ColumnName(name: "PredictedLabel")]
        public bool TurnOnHeat { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
