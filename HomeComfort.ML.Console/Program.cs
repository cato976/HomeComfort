using System;
using System.Collections.Generic;
using Microsoft.ML;
using HomeComfort.ML;
using HomeComfort.ML.Model;
//using HomeComfortML.Model;

namespace HomeComfort.ML.Console
{
    class Program
    {
        static List<FeedbackTrainingData> trainingData = new List<FeedbackTrainingData>();
        static List<FeedbackTrainingData> testData = new List<FeedbackTrainingData>();

        static void Main(string[] args)
        {
            // Step 1 :- We need to load the training data
            //trainingData = SprinklerLab.LoadTrainingData(trainingData);
            trainingData.AddRange(ComfortLab.ReadDataForPassXDays());
            //System.Console.WriteLine(trainingData.Count);
            //SprinklerLab.LoadTrainingDataFromFile(trainingData);

            // Step 2 :- Create object of MLContext
            var mlContext = new MLContext();

            // Step 3 :- Convert your data in to IDataView
            IDataView dataView = mlContext.Data.LoadFromEnumerable<FeedbackTrainingData>(trainingData);

            var model = ComfortLab.TrainMachine(mlContext, dataView);
            var modelHeat = ComfortLab.TrainHeatMachine(mlContext, dataView);

            //// Step 4 :- We need to create the pipeline and define the workflows in it.
            //var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(FeedbackTrainingData.TurnOnSprinklers))
            //    .Append(mlContext.Transforms.Concatenate("Features", nameof(FeedbackTrainingData.FeedbackRainfall)))
            //    .Append(mlContext.BinaryClassification.Trainers.FastTree(numberOfLeaves: 50, numberOfTrees: 50, minimumExampleCountPerLeaf: 1));

            //// Step 5 :- Train the algorithm and we want the model out
            //var model = pipeline.Fit(dataView);

            // Step 6 :- Load the test data and run the test data to check the models accuracy
            testData = ComfortLab.LoadTestData(testData);
            testData = ComfortLab.LoadTestDataFromFile(testData);

            IDataView testDataView = mlContext.Data.LoadFromEnumerable<FeedbackTrainingData>(testData);

            var predictions = model.Transform(testDataView);
            var heatPredictions = modelHeat.Transform(testDataView);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            var heatMetrics = mlContext.BinaryClassification.Evaluate(heatPredictions, "Label");
            System.Console.WriteLine($"metrics accuracy: {metrics.Accuracy}");
            System.Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            System.Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            System.Console.WriteLine($"F1Score: {metrics.F1Score:P2}");

            System.Console.WriteLine(Environment.NewLine);
            System.Console.WriteLine($"Heat metrics accuracy: {heatMetrics.Accuracy}");
            System.Console.WriteLine($"Heat Accuracy: {heatMetrics.Accuracy:P2}");
            System.Console.WriteLine($"Heat Auc: {heatMetrics.AreaUnderRocCurve:P2}");
            System.Console.WriteLine($"Heat F1Score: {heatMetrics.F1Score:P2}");

            // Step 7 :- use the model
            string strcont = "Y";
            while (strcont == "Y")
            {
                System.Console.WriteLine("Enter outdoor temp.");
                string outdoor = System.Console.ReadLine().ToString();
                float output;
                float.TryParse(outdoor, out output);
                if (output == 0 && outdoor != "0")
                {
                    strcont = outdoor;
                }
                System.Console.WriteLine("Enter indoor temp.");
                string indoorTemperature = System.Console.ReadLine().ToString();
                System.Console.WriteLine("Enter time of day");
                string timeOfDay = System.Console.ReadLine().ToString();

                // Create single instance of sample data from first line of dataset for model input
                ComfortModelInput sampleData = new ComfortModelInput()
                {
                    OutdoorTemp = float.Parse(outdoor),
                    IndoorTemp = float.Parse(indoorTemperature),
                    IndoorHumidity = 0F,
                    TimeOfDay = ((DateTimeOffset)DateTime.Parse(timeOfDay)).ToUnixTimeSeconds()
                };

                // Make a single prediction on the sample data and print results
                var predictionResult = HeatConsumeModel.Predict(sampleData);

                System.Console.WriteLine(Environment.NewLine + "Using model to make single prediction -- Comparing actual TurnOnHeat with predicted TurnOnHeat from sample data...\n\n");
                System.Console.WriteLine($"OutdoorTemp: {sampleData.OutdoorTemp}");
                System.Console.WriteLine($"IndoorTemp: {sampleData.IndoorTemp}");
                System.Console.WriteLine($"IndoorHumidity: {sampleData.IndoorHumidity}");
                System.Console.WriteLine($"TimeOfDay: {sampleData.TimeOfDay}");
                bool heatOn = predictionResult.Score > .5 ? true : false;
                System.Console.WriteLine($"\n\nPredicted TurnOnHeat: {heatOn}\n\n");

                // Make a single prediction on the sample data and print results
                var predictionACResult = ACConsumeModel.Predict(sampleData);

                System.Console.WriteLine(Environment.NewLine + "Using model to make single prediction -- Comparing actual TurnOnAC with predicted TurnOnAC from sample data...\n\n");
                System.Console.WriteLine($"OutdoorTemp: {sampleData.OutdoorTemp}");
                System.Console.WriteLine($"IndoorTemp: {sampleData.IndoorTemp}");
                System.Console.WriteLine($"IndoorHumidity: {sampleData.IndoorHumidity}");
                System.Console.WriteLine($"TimeOfDay: {sampleData.TimeOfDay}");
                bool ACOn = predictionACResult.Score > .5 ? true : false;
                System.Console.WriteLine($"\n\nPredicted TurnOnAC: {ACOn}\n\n");

                var predictionFunction = mlContext.Model.CreatePredictionEngine<FeedbackTrainingData, FeedbackPrediction>(model);
                var predictionFunction2 = mlContext.Model.CreatePredictionEngine<FeedbackTrainingData, FeedbackHeatPrediction>(modelHeat);

                var feedbackInput = new FeedbackTrainingData();
                try
                {
                    feedbackInput.OutdoorTemp = float.Parse(outdoor);
                    feedbackInput.IndoorTemp = float.Parse(indoorTemperature);
                    feedbackInput.TimeOfDay = ((DateTimeOffset)DateTime.Parse(timeOfDay)).ToUnixTimeSeconds();
                    var feedbackPredicted = predictionFunction.Predict(feedbackInput);
                    System.Console.WriteLine(Environment.NewLine + $"TurnOnAC: {feedbackPredicted.TurnOnAC} | Prediction: {(System.Convert.ToBoolean(feedbackPredicted.TurnOnAC) ? "Positive" : "Negative")} | Probability: {feedbackPredicted.Probability} ");
                    System.Console.WriteLine($"TurnOnAC :- {feedbackPredicted.TurnOnAC}");
                    var feedbackPredicted2 = predictionFunction2.Predict(feedbackInput);
                    System.Console.WriteLine(Environment.NewLine + $"TurnOnHeat: {feedbackPredicted2.TurnOnHeat} | Prediction: {(System.Convert.ToBoolean(feedbackPredicted2.TurnOnHeat) ? "Positive" : "Negative")} | Probability: {feedbackPredicted2.Probability} ");
                    System.Console.WriteLine($"TurnOnHeat :- {feedbackPredicted2.TurnOnHeat}");
                    System.Console.WriteLine(Environment.NewLine);
                    System.Console.WriteLine("=============== End of process, hit <CTRL-C key to finish ===============");
                }
                catch { }
            }
        }
    }
}
