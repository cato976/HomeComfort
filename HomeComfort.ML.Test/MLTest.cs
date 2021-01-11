using System.Net.Http;
using NUnit.Framework;

namespace HomeComfort.ML.Test
{
    public class Tests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void ReadWeatherFile()
        {
            HttpClient httpClient = new HttpClient();

            var data = ComfortLab.ReadDataForPassXDays();
            Assert.AreEqual(44, data.Count);
        }
    }
}
