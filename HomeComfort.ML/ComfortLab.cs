using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using ChoETL;

namespace HomeComfort.ML
{
    public class ComfortLab
    {
        public static List<FeedbackTrainingData> LoadTrainingData(List<FeedbackTrainingData> trainingData)
        {
            trainingData.Add(new FeedbackTrainingData()
            {
                OutdoorTemp = 85F,
                IndoorTemp = 80F,
                IndoorHumidity = 70,
                TimeOfDay = ((DateTimeOffset)new DateTime(2020, 7, 25, 8, 0, 0)).ToUnixTimeSeconds(),
                TurnOnHeat = false,
                TurnOnAC = true
            });

            return trainingData;
        }

        public static List<FeedbackTrainingData> LoadTestData(List<FeedbackTrainingData> testData)
        {
            testData.Add(new FeedbackTrainingData()
            {
                OutdoorTemp = 83.5F,
                IndoorTemp = 79.0F,
                IndoorHumidity = 70,
                TimeOfDay = ((DateTimeOffset)new DateTime(2020, 7, 25, 10, 0, 0)).ToUnixTimeSeconds(),
                TurnOnAC = true,
                TurnOnHeat = false
            });
            return testData;
        }

        public static TransformerChain<BinaryPredictionTransformer<Microsoft.ML.Calibrators.CalibratedModelParametersBase<Microsoft.ML.Trainers.FastTree.FastTreeBinaryModelParameters,
               Microsoft.ML.Calibrators.PlattCalibrator>>> TrainMachine(MLContext mlContext, IDataView dataView)
        {
            // Step 4 :- We need to create the pipeline and define the workflows in it.
            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(FeedbackTrainingData.TurnOnAC))
                .Append(mlContext.Transforms.Concatenate("Features", nameof(FeedbackTrainingData.OutdoorTemp), nameof(FeedbackTrainingData.IndoorTemp), 
                nameof(FeedbackTrainingData.IndoorHumidity), nameof(FeedbackTrainingData.TimeOfDay)))
                .Append(mlContext.BinaryClassification.Trainers.FastTree(numberOfLeaves: 50, numberOfTrees: 50, minimumExampleCountPerLeaf: 1));


            // Step 5 :- Train the algorithm and we want the model out
            var model = pipeline.Fit(dataView);

            return model;
        }

        public static TransformerChain<BinaryPredictionTransformer<Microsoft.ML.Calibrators.CalibratedModelParametersBase<Microsoft.ML.Trainers.FastTree.FastTreeBinaryModelParameters,
               Microsoft.ML.Calibrators.PlattCalibrator>>> TrainHeatMachine(MLContext mlContext, IDataView dataView)
        {
            // Step 4 :- We need to create the pipeline and define the workflows in it.
            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(FeedbackTrainingData.TurnOnHeat))
                .Append(mlContext.Transforms.Concatenate("Features", nameof(FeedbackTrainingData.OutdoorTemp), nameof(FeedbackTrainingData.IndoorTemp), 
                nameof(FeedbackTrainingData.IndoorHumidity), nameof(FeedbackTrainingData.TimeOfDay)))
                .Append(mlContext.BinaryClassification.Trainers.FastTree(numberOfLeaves: 50, numberOfTrees: 50, minimumExampleCountPerLeaf: 1));

            // Step 5 :- Train the algorithm and we want the model out
            var model = pipeline.Fit(dataView);

            return model;
        }

        public static List<FeedbackTrainingData> LoadTestDataFromFile(List<FeedbackTrainingData> testData)
        {
            var data = ReadDataForPassXDays();
            var count = 50;
            for (int i = 0; i < count; i++)
            {
                testData.Add(new FeedbackTrainingData()
                {
                    OutdoorTemp = data[i].OutdoorTemp,
                    IndoorTemp = data[i].IndoorTemp,
                    IndoorHumidity = data[i].IndoorHumidity,
                    TimeOfDay = data[i].TimeOfDay,
                    TurnOnAC = data[i].TurnOnAC,
                    TurnOnHeat = data[i].TurnOnHeat
                });
            }
            testData.Add(new FeedbackTrainingData()
            {
                OutdoorTemp = 44F,
                IndoorTemp = 68,
                IndoorHumidity = 55,
                TimeOfDay = ((DateTimeOffset)new DateTime(2020, 12, 15, 17, 0, 0)).ToUnixTimeSeconds(),
                TurnOnAC = false,
                TurnOnHeat = true
            });

            return testData;
        }

        public static List<FeedbackTrainingData> ReadDataForPassXDays()
        {
            List<FeedbackTrainingData> trainingData = new List<FeedbackTrainingData>();
            String path = "c:\\temp\\WeatherData.txt";

            using (StreamReader sr = File.OpenText(path))
            {
                Random rnd = new Random();
                string s = "";
                while ((s = sr.ReadToEnd()) != string.Empty)
                {
                    if (!string.IsNullOrEmpty(s))
                    {
                        JArray weatherEntries = JArray.Parse(s);
                        foreach (JObject weatherEntry in weatherEntries)
                        {
                            var rainfall = (double)weatherEntry["daily"]["data"][0]["precipIntensityMax"];
                            var temperature = (double)weatherEntry["daily"]["data"][0]["temperatureHigh"];
                            var indoorTemperature = (float)weatherEntry["daily"]["data"][0]["temperatureLow"];
                            var windSpeed = (double)weatherEntry["daily"]["data"][0]["windSpeed"];
                            var time = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Local).AddSeconds(long.Parse(weatherEntry["daily"]["data"][0]["time"].ToString()));
                            time = time.AddHours(rnd.Next(14));
                            //Console.WriteLine($"rainfall: {rainfall}");
                            //Console.WriteLine($"temperature: {temperature}");
                            var trfl = rainfall < 1 && temperature > 70 && windSpeed < 5 ? true : false;
                            //Console.WriteLine($"TurnOnSprinklers : {trfl}");
                            float indoorTemp = indoorTemperature;
                            trainingData.Add(new FeedbackTrainingData()
                            {
                                IndoorTemp = indoorTemp,
                                OutdoorTemp = (float)temperature,
                                TimeOfDay = ((DateTimeOffset)time).ToUnixTimeSeconds(),
                                TurnOnAC = temperature > 76 && indoorTemp > 74 && IsBetweenTime(time, new TimeSpan(6, 0, 0), new TimeSpan(23, 0, 0)) ? true : false,
                                TurnOnHeat = temperature < 73 && indoorTemp < 72 && IsBetweenTime(time, new TimeSpan(6, 0, 0), new TimeSpan(23, 0, 0)) ? true : false
                            });
                        }
                    }
                    //Console.WriteLine(s);
                }

                trainingData.Add(new FeedbackTrainingData()
                {
                    OutdoorTemp = 85F,
                    IndoorTemp = 80F,
                    IndoorHumidity = 70,
                    TimeOfDay = ((DateTimeOffset)new DateTime(2020, 7, 25, 8, 0, 0)).ToUnixTimeSeconds(),
                    TurnOnHeat = false,
                    TurnOnAC = true
                });

                //String ComfortPath = "c:\\temp\\ComfortrData.json";
                //String ComfortPathCSV = "c:\\temp\\ComfortrData.csv";

                //using (var r = new ChoJSONReader(ComfortPath))
                //{
                //    using (var w = new ChoCSVWriter(ComfortPathCSV).WithFirstLineHeader())
                //    {
                //        w.Write(r);
                //    }
                //}

                //String ComfortPath = "c:\\temp\\ComfortrData.json";
                //if (!File.Exists(ComfortPath))
                //{
                //    // Create a file to write to.
                //    using (StreamWriter sw = File.CreateText(ComfortPath))
                //    {
                //        sw.WriteLine("[");
                //    }
                //}

                //foreach (var comfort in trainingData)
                //{
                //    // Add the training data to another file
                //    using (StreamWriter sw = File.AppendText(ComfortPath))
                //    {
                //        var json = Newtonsoft.Json.JsonConvert.SerializeObject(comfort);
                //        sw.WriteLine(json);
                //        sw.WriteLine(",");
                //    }
                //}

                //using (StreamWriter sw = File.AppendText(ComfortPath))
                //{
                //    sw.WriteLine("]");
                //}

                return trainingData;
            }
        }

        private static bool IsBetweenTime(DateTime dateTime, TimeSpan start, TimeSpan end)
        {
            // convert datetime to a TimeSpanT
            TimeSpan now = dateTime.TimeOfDay;
            // see if start comes before end
            if (start < end)
            {
                return start <= now && now <= end;
            }
            return !(end < now && now < start);
        }
    }
}
