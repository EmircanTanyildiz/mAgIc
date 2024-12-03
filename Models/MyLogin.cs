using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace mAgIc.Models
{
    public class MyLogin
    {
        public string email {  get; set; }
        public string password { get; set; }
        public virtual ICollection<UserActions> UserActions { get; set; }
    }
}