using System;
using HalconDotNet;

namespace HalconWeek11
{
    internal static class HalconBatchDemo
    {
        private static int Inspect(string imageName, out int objectCount)
        {
            HObject image = null;
            HObject region = null;
            HObject connected = null;
            HObject selected = null;
            objectCount = 0;

            try
            {
                HOperatorSet.ReadImage(out image, imageName);
                HOperatorSet.Threshold(image, out region, 180, 255);
                HOperatorSet.Connection(region, out connected);
                HOperatorSet.SelectShape(connected, out selected, "area", "and", 80, 999999);

                HTuple count;
                HOperatorSet.CountObj(selected, out count);
                objectCount = count.I;
                return objectCount > 0 ? 0 : 10; // 0=OK, 10=NO_OBJECT
            }
            catch (HalconException ex)
            {
                Console.Error.WriteLine("HALCON error: " + ex.Message);
                return 20; // PROCESSING_ERROR
            }
            finally
            {
                if (selected != null) selected.Dispose();
                if (connected != null) connected.Dispose();
                if (region != null) region.Dispose();
                if (image != null) image.Dispose();
            }
        }

        private static void Main()
        {
            string[] images = { "fabrik", "mreut", "rings_and_nuts" };
            foreach (string image in images)
            {
                int count;
                int status = Inspect(image, out count);
                Console.WriteLine("{0}: status={1}, count={2}", image, status, count);
            }
        }
    }
}
