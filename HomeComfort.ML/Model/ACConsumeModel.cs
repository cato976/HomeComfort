// This file was auto-generated by ML.NET Model Builder. 

using System;
using System.IO;
using Microsoft.ML;

namespace HomeComfort.ML.Model
{
    public class ACConsumeModel
    {
        private static Lazy<PredictionEngine<ComfortModelInput, ACModelOutput>> PredictionEngine = new Lazy<PredictionEngine<ComfortModelInput, ACModelOutput>>(CreatePredictionEngine);

        public static string MLNetModelPath = Path.GetFullPath("ACMLModel.zip");

        // For more info on consuming ML.NET models, visit https://aka.ms/mlnet-consume
        // Method for consuming model in your app
        public static ACModelOutput Predict(ComfortModelInput input)
        {
            ACModelOutput result = PredictionEngine.Value.Predict(input);
            return result;
        }

        public static PredictionEngine<ComfortModelInput, ACModelOutput> CreatePredictionEngine()
        {
            // Create new MLContext
            MLContext mlContext = new MLContext();

            // Load model & create prediction engine
            ITransformer mlModel = mlContext.Model.Load(MLNetModelPath, out var modelInputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<ComfortModelInput, ACModelOutput>(mlModel);

            return predEngine;
        }
    }
}
