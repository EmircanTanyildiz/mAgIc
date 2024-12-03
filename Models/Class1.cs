using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Web;

namespace mAgIc.Models
{
    public class Class1
    {
    }
    public class UserAction
    {
        public int ID { get; set; }
        public int IDYapan { get; set; }
        public string yuklenenVideoPath { get; set; } 
        public DateTime Zaman { get; set; }
        public string TahminEdilenKategori { get; set; }
        [ForeignKey("IDYapan")]
        public virtual Users User { get; set; }
    }

}