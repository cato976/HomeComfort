using System;
using Microsoft.ML.Data;

namespace HomeComfort.ML
{
    public class FeedbackTrainingData
    {
        [LoadColumn(0)]
        public float OutdoorTemp { get; set; }
    
        [LoadColumn(1)]
        public float IndoorTemp { get; set; }

        [LoadColumn(2)]
        public float IndoorHumidity { get; set; }

        [LoadColumn(3)]
        public float TimeOfDay { get; set; }

        [LoadColumn(4)]
        public bool TurnOnAC { get; set; }

        [LoadColumn(5)]
        public bool TurnOnHeat { get; set; }
    }
}
