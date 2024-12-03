using mAgIc.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;

namespace mAgIc.Controllers
{
    public class AccountController : Controller
    {
        mAgIcProjectEntities db = new mAgIcProjectEntities();
        // GET: Account
        public ActionResult Index()
        {
            return View();
        }
        [HttpGet]
        public ActionResult Register()
        {
            return View();
        }
        [HttpPost]
        public ActionResult Register(Users ur)
        {
            if (ModelState.IsValid)
            {
                if (db.Users.Any(x => x.email == ur.email))
                {
                    ViewBag.Message = "Bu E Mail Zaten Kayıtlı";
                }
                else
                {
                    db.Users.Add(ur);
                    db.SaveChanges();
                    Response.Write("<script>alert('Registration Succesful')</script>");
                }
            }
            return View();
        }
        [HttpGet]
        public ActionResult Login()
        {
            return View();
        }

        [HttpPost]
        public ActionResult Login(MyLogin l)
        {
            var query = db.Users.SingleOrDefault(m => m.email == l.email && m.password == l.password);
            if (query != null)
            {
                // Kullanıcı ID'sini session'da sakla
                Session["UserId"] = query.ID; // Users tablosundaki ID alanı
                Response.Write("<script>alert('Login Successful!')</script>");
                return RedirectToAction("Upload", "Video");
            }
            else
            {
                Response.Write("<script>alert('Error: Invalid credentials')</script>");
            }
            return View();
        }
    }
}