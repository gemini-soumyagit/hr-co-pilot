
export default async ({ req, res, log, error }) => {
  if(req.metho.get){

    return res.send("welcome to era of AI !")
  }
  if(req.metho.post){

    return res.json({
      'sendData':req.body,
      'googleAPiKey': process.env.GOOGLE_GEMINI_API_KEY
    })
  }

};
