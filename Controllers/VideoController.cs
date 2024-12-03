using System;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Web;
using System.Web.Mvc;
using mAgIc.Models;
using Newtonsoft.Json;
using System.Text;

namespace mAgIc.Controllers
{
    public class VideoController : Controller
    {
        mAgIcProjectEntities _context = new mAgIcProjectEntities();

        [HttpGet]
        public ActionResult Upload()
        {
            if (Session["UserId"] == null)
            {
                return RedirectToAction("Login", "Account");
            }
            return View();
        }

        [HttpPost]
        public async System.Threading.Tasks.Task<ActionResult> Upload(HttpPostedFileBase videoFile)
        {
            if (Session["UserId"] == null)
            {
                return RedirectToAction("Login", "Account");
            }

            if (videoFile != null && videoFile.ContentLength > 0)
            {
                var gelenID = Session["UserId"].ToString();

                
                string videoPath = Path.GetFileName(videoFile.FileName);
                string serverPath = Server.MapPath("~/UploadedVideos/");
                if (!Directory.Exists(serverPath))
                {
                    Directory.CreateDirectory(serverPath);
                }
                string fullPath = Path.Combine(serverPath, videoPath);
                videoFile.SaveAs(fullPath);

                
                string formattedPath = FormatVideoPath(fullPath);

                
                string predictedCategory = await PostVideoPathToAPI(formattedPath);

                
                int userId = (int)Session["UserId"];
                var userAction = new UserActions
                {
                    IDYapan = userId,
                    yuklenenVideoPath = formattedPath,
                    Zaman = DateTime.Now,
                    tahminEdilenKategori = predictedCategory
                };

                try
                {
                    _context.UserActions.Add(userAction);
                    _context.SaveChanges();
                    ViewBag.Message = predictedCategory;
                }
                catch (System.Data.Entity.Validation.DbEntityValidationException ex)
                {
                    StringBuilder validationErrors = new StringBuilder();

                    
                    foreach (var validationErrorsList in ex.EntityValidationErrors)
                    {
                        foreach (var validationError in validationErrorsList.ValidationErrors)
                        {
                            validationErrors.AppendLine($"Property: {validationError.PropertyName}, Error: {validationError.ErrorMessage}");
                        }
                    }

                    
                    ViewBag.ValidationErrorDetails = validationErrors.ToString();

                    
                    Console.WriteLine(validationErrors.ToString());

                    
                    ViewBag.Message = "Veritabanı doğrulama hatası oluştu.";
                    return View();
                }
            }
            else
            {
                ViewBag.Message = "Lütfen bir video seçin.";
            }

            return View();
        }

        
        private string FormatVideoPath(string path)
        {
            return path.Replace("\\", "\\\\");
        }

        
        private async System.Threading.Tasks.Task<string> PostVideoPathToAPI(string videoPath)
        {
            using (HttpClient client = new HttpClient())
            {
                string apiUrl = "http://localhost:5000/predict";

                var postData = new
                {
                    video_path = videoPath
                };

                string json = JsonConvert.SerializeObject(postData);
                StringContent content = new StringContent(json, Encoding.UTF8, "application/json");

                HttpResponseMessage response = await client.PostAsync(apiUrl, content);

                if (response.IsSuccessStatusCode)
                {
                    string responseBody = await response.Content.ReadAsStringAsync();
                    dynamic result = JsonConvert.DeserializeObject(responseBody);

                    
                    return result?.predicted_class ?? "Bilinmiyor";
                }
                else
                {
                    throw new Exception("API'ye bağlanırken hata oluştu: " + response.StatusCode);
                }
            }
        }

    }
}
